                           return None                                                        except Exception as e:                                                     if not isinstance(e, RETRYABLE_EXCEPTIONS):                                 log_error(f"Unexpected non-retryable error calculating position size: {e}", exc_info=True)                                               # If it was a retryable error, the decorator would have logged and raised it.                                                                 return None
                                                                                                                                              # --- Order Placement Functions (Decorated) ---                                                                                               # Helper for cancellation with retry                                   @api_retry_decorator                                                   def cancel_order_with_retry(exchange_instance: ccxt.Exchange, order_id: str, trading_symbol: str):                                                """Cancels an order by ID with retry logic."""                         if not order_id:                                                           log_debug("No order ID provided to cancel.")                           return # Nothing to cancel                                                                                                                log_info(f"Attempting to cancel order ID: {order_id} for {trading_symbol}...")                                                                if not SIMULATION_MODE:                                                    exchange_instance.cancel_order(order_id, trading_symbol)               log_info(f"Order {order_id} cancellation request sent.")           else:                                                                      log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")                                                                                                                                                                                                                    @api_retry_decorator                                                   def place_market_order(exchange_instance: ccxt.Exchange, trading_symbol: str, side: str, amount: float, reduce_only: bool = False) -> Optional[Dict[str, Any]]:
    """Places a market order ('buy' or 'sell') with retry logic. Optionally reduceOnly."""                                                        if side not in ['buy', 'sell']:                                            log_error(f"Invalid order side: '{side}'.")                            return None                                                        if amount <= 0:                                                            log_error(f"Invalid order amount: {amount}.")                          return None                                                                                                                               try:                                                                       market = exchange_instance.market(trading_symbol)                      base_currency: str = market.get('base', '')                            quote_currency: str = market.get('quote', '')                          amount_formatted = float(exchange_instance.amount_to_precision(trading_symbol, amount))
                                                                               log_info(f"Attempting to place {side.upper()} market order for {amount_formatted:.{amount_precision_digits}f} {base_currency} {'(ReduceOnly)' if reduce_only else ''}...")                                                                                                                  params = {}                                                            # Check if exchange explicitly supports reduceOnly for market orders, common for futures/swap
        if reduce_only and exchange_instance.has.get('reduceOnly'):                 # Check if market type likely supports it (swap/future)                if market.get('swap') or market.get('future') or market.get('contract'):
                  params['reduceOnly'] = True                                            log_debug("Applying 'reduceOnly=True' to market order.")                                                                                 else:                                                                       log_warning("ReduceOnly requested but market type is likely SPOT. Ignoring reduceOnly param for market order.")                                                                                                                                                                   if SIMULATION_MODE:                                                        log_warning("!!! SIMULATION: Market order placement skipped.")                                                                                # Create a realistic dummy order response                              sim_price = 0.0                                                        try: # Fetch ticker safely                                                  # Fetching ticker can also fail, apply retry decorator? Maybe overkill here.
                 ticker = exchange_instance.fetch_ticker(trading_symbol)                                                                                       sim_price = ticker['last'] if ticker and 'last' in ticker else 0.0                                                                       except Exception as ticker_e:                                              log_warning(f"Could not fetch ticker for simulation price: {ticker_e}")                                                                                                                                          order_id = f'sim_market_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'                                                                      sim_cost = amount_formatted * sim_price if sim_price > 0 else 0.0
            order = {                                                                  'id': order_id, 'clientOrderId': order_id, 'timestamp': int(time.time() * 1000),                                                              'datetime': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'closed', # Assume instant fill                                                 'symbol': trading_symbol, 'type': 'market', 'timeInForce': 'IOC',                                                                             'side': side, 'price': sim_price, 'average': sim_price,                'amount': amount_formatted, 'filled': amount_formatted, 'remaining': 0.0,                                                                     'cost': sim_cost, 'fee': None, 'info': {'simulated': True, 'reduceOnly': params.get('reduceOnly', False)}                                 }
        else:                                                                      # --- LIVE TRADING ---
            log_warning(f"!!! LIVE MODE: Placing real market order {'(ReduceOnly)' if params.get('reduceOnly') else ''}.")                                order = exchange_instance.create_market_order(trading_symbol, side, amount_formatted, params=params)                                          # --- END LIVE TRADING ---                                                                                                                log_info(f"{side.capitalize()} market order request processed {'(Simulated)' if SIMULATION_MODE else ''}.")                                   order_id: Optional[str] = order.get('id')                              order_status: Optional[str] = order.get('status')                      # Use 'average' if available (filled price), fallback to 'price' (less reliable for market)                                                   order_price: Optional[float] = order.get('average', order.get('price'))                                                                       order_filled: Optional[float] = order.get('filled')                    order_cost: Optional[float] = order.get('cost')                
        price_str: str = f"{order_price:.{price_precision_digits}f}" if isinstance(order_price, (int, float)) else "N/A"                              filled_str: str = f"{order_filled:.{amount_precision_digits}f}" if isinstance(order_filled, (int, float)) else "N/A"                          cost_str: str = f"{order_cost:.{price_precision_digits}f}" if isinstance(order_cost, (int, float)) else "N/A"                                                                                                        log_info(f"Order Result | ID: {order_id or 'N/A'}, Status: {order_status or 'N/A'}, Avg Price: {price_str}, "                                             f"Filled: {filled_str} {base_currency}, Cost: {cost_str} {quote_currency}")                                                       # Add short delay after placing order to allow exchange processing / state update                                                             time.sleep(1.5) # Increased delay slightly                             return order                                                                                                                              # Handle specific non-retryable errors
    except ccxt.InsufficientFunds as e:                                        log_error(f"Insufficient funds for {side} {amount} {trading_symbol}. Error: {e}")                                                             return None                                                        except ccxt.OrderNotFound as e: # Can happen if order rejected immediately                                                                        log_error(f"OrderNotFound error placing {side} {amount} {trading_symbol}. Likely rejected immediately. Error: {e}")                           return None                                                        except ccxt.InvalidOrder as e:
         log_error(f"Invalid market order parameters for {side} {amount} {trading_symbol}: {e}")                                                       return None                                                       except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors                                                                       log_error(f"Exchange specific error placing {side} market order for {trading_symbol}: {e}")                                                   return None                                                        except Exception as e:                                                     if not isinstance(e, RETRYABLE_EXCEPTIONS):                                 log_error(f"Unexpected non-retryable error placing {side} market order: {e}", exc_info=True)                                             return None                                                                                                                                                                                                  # Note: Applying retry to the *entire* SL/TP function can be complex if one order                                                             # succeeds and the other fails, leading to duplicate orders on retry.  # A more granular approach (retrying individual create_order calls within) might be better,                                                   # but for simplicity, we apply it to the whole function here. Be cautious in live mode.
@api_retry_decorator                                                   def place_sl_tp_orders(exchange_instance: ccxt.Exchange, trading_symbol: str, position_side: str, quantity: float, sl_price: float, tp_price: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:       """                                                                    Places stop-loss (stop-market preferred) and take-profit (limit) orders.                                                                      Uses reduceOnly. Retries the whole placement process on network errors.                                                                       """                                                                    sl_order: Optional[Dict[str, Any]] = None                              tp_order: Optional[Dict[str, Any]] = None                                                                                                     if quantity <= 0 or position_side not in ['long', 'short']:                log_error(f"Invalid input for SL/TP placement: Qty={quantity}, Side='{position_side}'")                                                       return None, None                                                                                                                         try:                                                                       market = exchange_instance.market(trading_symbol)                      close_side = 'sell' if position_side == 'long' else 'buy'              qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, quantity))                                                        sl_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, sl_price))                                                    tp_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, tp_price))                                                                                                                           # Check exchange capabilities                                          # ReduceOnly is often a param, not a separate capability check in ccxt `has`                                                                  has_reduce_only_param = True # Assume most modern exchanges support it in params                                                              has_stop_market = exchange_instance.has.get('createStopMarketOrder', False)                                                                   has_stop_limit = exchange_instance.has.get('createStopLimitOrder', False)                                                                     has_limit = exchange_instance.has.get('createLimitOrder', True) # Assume basic limit exists
                                                                               sl_params = {'reduceOnly': True} if has_reduce_only_param else {}                                                                             tp_params = {'reduceOnly': True} if has_reduce_only_param else {}                                                                     
        # --- Place Stop-Loss Order ---                                        log_info(f"Attempting SL order: {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} @ trigger ~{sl_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in sl_params else ''}")
        sl_order_type = None                                                   sl_order_price = None # Only needed for limit types            
        # Prefer stopMarket if available via createOrder                       # Note: Some exchanges require create_stop_market_order explicitly                                                                            if has_stop_market or 'stopMarket' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):                                      sl_order_type = 'stopMarket'                                           # CCXT standard uses 'stopPrice' in params for stop orders via create_order
            sl_params['stopPrice'] = sl_price_formatted                            log_debug("Using stopMarket type for SL via createOrder.")         elif has_stop_limit or 'stopLimit' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):                                      sl_order_type = 'stopLimit'                                            # StopLimit requires a trigger (stopPrice) and a limit price.                                                                                 # Set limit price slightly worse than trigger to increase fill chance                                                                         limit_offset = abs(tp_price_formatted - sl_price_formatted) * 0.1 # 10% of TP-SL range as offset                                              limit_offset = max(limit_offset, min_tick * 5) # Ensure offset is at least a few ticks                                                        sl_limit_price = sl_price_formatted - limit_offset if close_side == 'sell' else sl_price_formatted + limit_offset                             sl_order_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price))                                                                                                                         sl_params['stopPrice'] = sl_price_formatted                            sl_params['price'] = sl_order_price # The limit price for the order placed after trigger                                                      log_warning(f"Using stopLimit type for SL. Trigger: {sl_price_formatted:.{price_precision_digits}f}, Limit: {sl_order_price:.{price_precision_digits}f}.")                                                       else:
            log_error(f"Exchange {exchange_instance.id} supports neither stopMarket nor stopLimit orders via CCXT createOrder. Cannot place automated SL.")                                                                      # Continue to try placing TP                               
        if sl_order_type:                                                          if SIMULATION_MODE:                                                        log_warning("!!! SIMULATION: Stop-loss order placement skipped.")                                                                             sl_order_id = f'sim_sl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'                                                                       sl_order = {
                    'id': sl_order_id, 'status': 'open', 'symbol': trading_symbol,                                                                                'type': sl_order_type, 'side': close_side, 'amount': qty_formatted,                                                                           'price': sl_order_price, # Limit price if stopLimit                    'stopPrice': sl_params.get('stopPrice'), # Trigger price
                    'average': None, 'filled': 0.0, 'remaining': qty_formatted, 'cost': 0.0,
                    'info': {'simulated': True, 'reduceOnly': sl_params.get('reduceOnly', False)}                                                             }                                                                  else:                                                                      # --- LIVE TRADING ---
                log_warning(f"!!! LIVE MODE: Placing real {sl_order_type} stop-loss order.")                                                                  # Use create_order as it handles various stop types with params                                                                               sl_order = exchange_instance.create_order(                                 symbol=trading_symbol,                                                 type=sl_order_type,                                                    side=close_side,                                                       amount=qty_formatted,                                                  price=sl_order_price, # Required only for limit types (like stopLimit)                                                                        params=sl_params                                                   )                                                                      log_info(f"Stop-loss order request processed. ID: {sl_order.get('id', 'N/A')}")
                time.sleep(0.75) # Small delay after live order                        # --- END LIVE TRADING ---                                                                                                            # --- Place Take-Profit Order ---                                      log_info(f"Attempting TP order: {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} @ limit {tp_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in tp_params else ''}")                                                                            tp_order_type = 'limit' # Standard limit order for TP                                                                                         if has_limit: # Check if limit orders are supported (should almost always be true)                                                                 if SIMULATION_MODE:                                                        log_warning("!!! SIMULATION: Take-profit order placement skipped.")                                                                           tp_order_id = f'sim_tp_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                 tp_order = {                                                               'id': tp_order_id, 'status': 'open', 'symbol': trading_symbol,                                                                                'type': tp_order_type, 'side': close_side, 'amount': qty_formatted,                                                                           'price': tp_price_formatted, 'average': None,
                     'filled': 0.0, 'remaining': qty_formatted, 'cost': 0.0,
                     'info': {'simulated': True, 'reduceOnly': tp_params.get('reduceOnly', False)}
                 }                                                                  else:                                                                     # --- LIVE TRADING ---                                                  log_warning(f"!!! LIVE MODE: Placing real {tp_order_type} take-profit order.")                                                                tp_order = exchange_instance.create_limit_order(                           symbol=trading_symbol,                                                 side=close_side,                                                       amount=qty_formatted,                                                  price=tp_price_formatted,                                              params=tp_params
                 )                                                                      log_info(f"Take-profit order request processed. ID: {tp_order.get('id', 'N/A')}")                                                             time.sleep(0.75) # Small delay after live order                        # --- END LIVE TRADING ---                                    else:                                                                       log_error(f"Exchange {exchange_instance.id} does not support limit orders. Cannot place take-profit.")                                        tp_order = None                                           
                                                                               # Final check and warnings if one failed                               sl_ok = sl_order and sl_order.get('id')                                tp_ok = tp_order and tp_order.get('id')                                if sl_ok and not tp_ok: log_warning("SL placed, but TP failed.")
        elif not sl_ok and tp_ok: log_warning("TP placed, but SL failed. Position unprotected!")                                                      elif not sl_ok and not tp_ok: log_error("Both SL and TP order placements failed.")                                                                                                                                   return sl_order, tp_order                                                                                                                 # Handle specific non-retryable errors
    except ccxt.InvalidOrder as e:
        log_error(f"Invalid order parameters placing SL/TP: {e}")              # Return whatever might have succeeded before the error
        return sl_order, tp_order                                          except ccxt.InsufficientFunds as e: # Should not happen with reduceOnly, but possible                                                              log_error(f"Insufficient funds error during SL/TP placement (unexpected): {e}")                                                               return sl_order, tp_order
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error placing SL/TP orders: {e}")        return sl_order, tp_order                                          except Exception as e:                                                     if not isinstance(e, RETRYABLE_EXCEPTIONS):                                 log_error(f"Unexpected non-retryable error in place_sl_tp_orders: {e}", exc_info=True)                                                   # Return None, None to indicate total failure if an exception occurred                                                                        return None, None
                                                                                                                                              # --- Position and Order Check Function (Decorated) ---                @api_retry_decorator                                                   def check_position_and_orders(exchange_instance: ccxt.Exchange, trading_symbol: str) -> bool:
    """                                                                    Checks if tracked SL/TP orders are still open. If not, assumes filled,                                                                        cancels the other order, resets local state, and returns True. Retries on network errors.                                                     Uses fetch_open_orders.
    """                                                                    global position                                                        if position['status'] is None:                                             return False # No active position to check                                                                                                log_debug(f"Checking open orders vs local state for {trading_symbol}...")
    position_reset_flag = False                                            try:                                                                       # Fetch Open Orders for the specific symbol (retried by decorator)                                                                            open_orders = exchange_instance.fetch_open_orders(trading_symbol)                                                                             log_debug(f"Found {len(open_orders)} open orders for {trading_symbol}.")                                                                                                                                             sl_order_id = position.get('sl_order_id')                              tp_order_id = position.get('tp_order_id')                                                                                                     if not sl_order_id and not tp_order_id:                                     # This state can occur if SL/TP placement failed initially or after a restart                                                                 # without proper state restoration of order IDs.                       log_warning(f"In {position['status']} position but no SL/TP order IDs tracked. Cannot verify closure via orders.")                            # Alternative: Try fetching current position size from exchange here?                                                                         # `fetch_positions` is often needed for futures/swaps, but can be complex/inconsistent.                                                       # Sticking to order-based check for now.                               return False # Cannot determine closure reliably                                                                                         # Check if tracked orders are still present in the fetched open orders                                                                        open_order_ids = {order.get('id') for order in open_orders if order.get('id')}                                                                sl_found_open = sl_order_id in open_order_ids                          tp_found_open = tp_order_id in open_order_ids                                                                                                 log_debug(f"Tracked SL ({sl_order_id or 'None'}) open: {sl_found_open}. Tracked TP ({tp_order_id or 'None'}) open: {tp_found_open}.")                                                                                # --- Logic for State Reset ---                                        order_to_cancel_id = None                                              assumed_close_reason = None                                                                                                                   # If SL was tracked but is no longer open:                             if sl_order_id and not sl_found_open:                                      log_info(f"{NEON_YELLOW}Stop-loss order {sl_order_id} no longer open. Assuming position closed via SL.{RESET}")                               position_reset_flag = True                                             order_to_cancel_id = tp_order_id # Attempt to cancel the TP order                                                                             assumed_close_reason = "SL"                                                                                                               # Else if TP was tracked but is no longer open:                        elif tp_order_id and not tp_found_open:                                    log_info(f"{NEON_GREEN}Take-profit order {tp_order_id} no longer open. Assuming position closed via TP.{RESET}")                              position_reset_flag = True                                             order_to_cancel_id = sl_order_id # Attempt to cancel the SL order                                                                             assumed_close_reason = "TP"                                                                                                               # If a reset was triggered:                                            if position_reset_flag:                                                    # Attempt to cancel the other order using the retry helper             if order_to_cancel_id:                                                      log_info(f"Attempting to cancel leftover {'TP' if assumed_close_reason == 'SL' else 'SL'} order {order_to_cancel_id}...")                     try:                                                                       # Use the decorated cancel helper                                      cancel_order_with_retry(exchange_instance, order_to_cancel_id, trading_symbol)                                                            except ccxt.OrderNotFound:                                                 log_info(f"Order {order_to_cancel_id} was already closed/cancelled.")                                                                     except Exception as e:                                                     # Error logged by cancel_order_with_retry or its decorator                                                                                    log_error(f"Failed to cancel leftover order {order_to_cancel_id} after retries. Manual check advised.", exc_info=False) # Don't need full trace here                                                        else:                                                                       log_debug(f"No corresponding {'TP' if assumed_close_reason == 'SL' else 'SL'} order ID was tracked or provided to cancel.")                                                                                                                                                            log_info("Resetting local position state.")                            # Reset all relevant fields                                            position.update({                                                          'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,                                                                      'stop_loss': None, 'take_profit': None, 'entry_time': None,                                                                                   'sl_order_id': None, 'tp_order_id': None,
                # Reset trailing stop fields as well                                   'highest_price_since_entry': None, 'lowest_price_since_entry': None,
                'current_trailing_sl_price': None                                  })                                                                     save_position_state() # Save the reset state                           display_position_status(position, price_precision_digits, amount_precision_digits) # Show updated status                                      return True # Indicate position was reset                                                                                                 # If both orders are still open, or only one was tracked and it's still open                                                                  log_debug("Position state appears consistent with tracked open orders.")                                                                      return False # Position was not reset                                                                                                     # Handle specific non-retryable errors from fetch_open_orders if they bypass decorator                                                        except ccxt.AuthenticationError as e:                                       log_error(f"Authentication error checking position/orders: {e}")
         return False                                                      except ccxt.ExchangeError as e:                                            log_error(f"Exchange error checking position/orders: {e}")             return False                                                       except Exception as e:                                                     if not isinstance(e, RETRYABLE_EXCEPTIONS):                                 log_error(f"Unexpected non-retryable error checking position/orders: {e}", exc_info=True)                                                # Let retry handler manage retryable exceptions                        return False # Assume state is uncertain if error occurred                                                                            # --- Trailing Stop Loss Update Function (From Snippet 2, Adapted) --- def update_trailing_stop(exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, last_atr: float):                           """Checks and updates the trailing stop loss if conditions are met."""                                                                        global position
    if (position['status'] is None or
        not enable_trailing_stop or                                            position['sl_order_id'] is None or # Need an active SL order to trail                                                                         last_atr <= 0): # Cannot trail if no position, TSL disabled, no SL order, or invalid ATR                                                      if enable_trailing_stop and position['status'] is not None and last_atr <= 0:                                                                     log_warning("Cannot perform trailing stop check: Invalid ATR value.")                                                                     elif enable_trailing_stop and position['status'] is not None and position['sl_order_id'] is None:                                                 log_warning("Cannot perform trailing stop check: No active SL order ID tracked.")                                                         return                                                                                                                                    log_debug(f"Checking trailing stop. Current SL Order: {position['sl_order_id']}, Current Tracked TSL Price: {position.get('current_trailing_sl_price')}")
                                                                           new_potential_tsl = None                                               activation_threshold_met = False                                       # Use current_trailing_sl_price if set, otherwise use the initial stop_loss from state                                                        current_effective_sl = position.get('current_trailing_sl_price') or position.get('stop_loss')                                                                                                                        if current_effective_sl is None:                                           log_warning("Cannot check TSL improvement: current effective SL price is None.")                                                              return                                                                                                                                    if position['status'] == 'long':                                           entry_price = position.get('entry_price', current_price) # Fallback to current if entry somehow missing                                       # Update highest price seen since entry                                highest_seen = position.get('highest_price_since_entry', entry_price)                                                                         if current_price > highest_seen:                                           position['highest_price_since_entry'] = current_price                  highest_seen = current_price                                           # Save state less frequently, e.g., only when TSL updates, to reduce disk I/O                                                                 # save_position_state() # Persist the new high - potentially too frequent                                                                                                                                        # Check activation threshold (price must move ATR * activation_multiplier above entry)                                                        activation_price = entry_price + (last_atr * trailing_stop_activation_atr_multiplier)                                                                                                                                if highest_seen > activation_price:                                        activation_threshold_met = True                                        # Calculate potential new trailing stop based on the highest price seen                                                                       new_potential_tsl = highest_seen - (last_atr * trailing_stop_atr_multiplier)                                                                  log_debug(f"Long TSL Activation Met. Highest: {highest_seen:.{price_precision_digits}f} > Activation: {activation_price:.{price_precision_digits}f}. Potential New TSL: {new_potential_tsl:.{price_precision_digits}f}")                                                                else:                                                                       log_debug(f"Long TSL Activation NOT Met. Highest: {highest_seen:.{price_precision_digits}f} <= Activation: {activation_price:.{price_precision_digits}f}")                                                                                                                                                                                                elif position['status'] == 'short':                                        entry_price = position.get('entry_price', current_price)               # Update lowest price seen since entry                                 lowest_seen = position.get('lowest_price_since_entry', entry_price)                                                                           if current_price < lowest_seen:                                            position['lowest_price_since_entry'] = current_price                   lowest_seen = current_price                                            # save_position_state() # Persist the new low - potentially too frequent                                                                                                                                         # Check activation threshold                                           activation_price = entry_price - (last_atr * trailing_stop_activation_atr_multiplier)                                                                                                                                if lowest_seen < activation_price:                                         activation_threshold_met = True                                        # Calculate potential new trailing stop                                new_potential_tsl = lowest_seen + (last_atr * trailing_stop_atr_multiplier)                                                                   log_debug(f"Short TSL Activation Met. Lowest: {lowest_seen:.{price_precision_digits}f} < Activation: {activation_price:.{price_precision_digits}f}. Potential New TSL: {new_potential_tsl:.{price_precision_digits}f}")                                                                 else:                                                                      log_debug(f"Short TSL Activation NOT Met. Lowest: {lowest_seen:.{price_precision_digits}f} >= Activation: {activation_price:.{price_precision_digits}f}")                                                
                                                                           if activation_threshold_met and new_potential_tsl is not None:             # --- Check if the new TSL is an improvement ---                       should_update_tsl = False                                              # Format potential new TSL to market precision for comparison          new_tsl_formatted = float(exchange_instance.price_to_precision(trading_symbol, new_potential_tsl))                                                                                                                   if position['status'] == 'long':
             # New TSL must be strictly higher than current effective SL                                                                                   # And must be below the current price (with a small buffer maybe?) to avoid immediate stop-out                                                if new_tsl_formatted > current_effective_sl and new_tsl_formatted < current_price:                                                                should_update_tsl = True                                               log_debug(f"Long TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) > Current ({current_effective_sl:.{price_precision_digits}f}) AND < Price ({current_price:.{price_precision_digits}f}) -> OK")                                                                         else:                                                                      log_debug(f"Long TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) vs Current ({current_effective_sl:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}) -> NO UPDATE")                                                                                                                                      elif position['status'] == 'short':
             # New TSL must be strictly lower than current effective SL             # And must be above the current price                                  if new_tsl_formatted < current_effective_sl and new_tsl_formatted > current_price:                                                                should_update_tsl = True                                               log_debug(f"Short TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) < Current ({current_effective_sl:.{price_precision_digits}f}) AND > Price ({current_price:.{price_precision_digits}f}) -> OK")                                                                        else:                                                                       log_debug(f"Short TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) vs Current ({current_effective_sl:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}) -> NO UPDATE")                                                                                                                                    if should_update_tsl:                                                      log_info(f"{NEON_YELLOW}Trailing Stop Update Triggered! New target SL: {new_tsl_formatted:.{price_precision_digits}f}{RESET}")
                                                                                   # --- Modify Existing SL Order (Cancel old, Place new) ---             old_sl_id = position['sl_order_id']                                    if not old_sl_id:                                                          log_error("TSL triggered but no old SL order ID found in state. Cannot modify.")                                                              return                                                                                                                                    try:                                                                       # 1. Cancel Old SL Order (using retry helper)                          cancel_order_with_retry(exchange_instance, old_sl_id, trading_symbol)                                                                         log_info(f"Old SL order {old_sl_id} cancellation request sent/simulated.")                                                                    # Short delay to allow cancellation processing                         time.sleep(1.0)                                                                                                                               # 2. Place New SL Order (use place_sl_tp_orders, but only use the SL part)                                                                    # We need the original TP price to calculate stopLimit offset if needed, get from state                                                       original_tp_price = position.get('take_profit', current_price * (1 + 0.05) if position['status'] == 'long' else current_price * (1 - 0.05)) # Fallback TP                                                            log_info(f"Placing new TSL order at {new_tsl_formatted:.{price_precision_digits}f}")                                                                                                                                 # Call place_sl_tp_orders but only care about the sl_order result                                                                             # Pass a dummy TP price far away, or the original TP if available                                                                             new_sl_order, _ = place_sl_tp_orders(                                      exchange_instance=exchange_instance,
                    trading_symbol=trading_symbol,                                         position_side=position['status'],                                      quantity=position['quantity'],                                         sl_price=new_tsl_formatted,                                            tp_price=original_tp_price # Pass original TP, place_sl_tp should only place SL if needed                                                     # NOTE: place_sl_tp_orders currently tries to place both. Need modification                                                                   # or a dedicated place_sl_order function.                              # Let's modify place_sl_tp_orders to accept optional placement flags.                                                                         # ---> Modification needed in place_sl_tp_orders or use direct create_order here.                                                                                                                                    # --- Alternative: Direct Placement (Simplified) ---                                                                                          # Replicating the SL placement logic here:
                )                                                                      new_sl_order_direct = None # Placeholder for direct placement result                                                                          try:                                                                       close_side = 'sell' if position['status'] == 'long' else 'buy'                                                                                qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, position['quantity']))                                            has_reduce_only_param = True
                    has_stop_market = exchange_instance.has.get('createStopMarketOrder', False)                                                                   has_stop_limit = exchange_instance.has.get('createStopLimitOrder', False)                                                                     new_sl_params = {'reduceOnly': True} if has_reduce_only_param else {}
                    new_sl_order_type = None                                               new_sl_order_price = None                                                                                                                     if has_stop_market or 'stopMarket' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):                                      new_sl_order_type = 'stopMarket'                                       new_sl_params['stopPrice'] = new_tsl_formatted                     elif has_stop_limit or 'stopLimit' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):                                      new_sl_order_type = 'stopLimit'                                        limit_offset = abs(original_tp_price - new_tsl_formatted) * 0.1                                                                               limit_offset = max(limit_offset, min_tick * 5)                         sl_limit_price = new_tsl_formatted - limit_offset if close_side == 'sell' else new_tsl_formatted + limit_offset                               new_sl_order_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price))                                              new_sl_params['stopPrice'] = new_tsl_formatted                         new_sl_params['price'] = new_sl_order_price                        else:
                         raise ccxt.NotSupported("Exchange supports neither stopMarket nor stopLimit for TSL.")                                                                                                                          log_info(f"Attempting direct placement of new TSL order ({new_sl_order_type})")                                                               if SIMULATION_MODE:                                                        log_warning("!!! SIMULATION: New TSL order placement skipped.")                                                                               new_sl_order_id = f'sim_tsl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'                                                                  new_sl_order_direct = {'id': new_sl_order_id, 'status': 'open', 'info': {'simulated': True}}                                              else:
                        log_warning(f"!!! LIVE MODE: Placing real {new_sl_order_type} TSL order.")                                                                    # Apply retry to this specific call                                    @api_retry_decorator                                                   def place_new_tsl_order_direct():                                          return exchange_instance.create_order(                                     symbol=trading_symbol, type=new_sl_order_type, side=close_side,                                                                               amount=qty_formatted, price=new_sl_order_price, params=new_sl_params                                                                      )                                                                  new_sl_order_direct = place_new_tsl_order_direct()                                                                                                                                                               # --- End Direct Placement ---                                                                                                                                                                                       if new_sl_order_direct and new_sl_order_direct.get('id'):                                                                                         new_id = new_sl_order_direct['id']                                     log_info(f"New trailing SL order placed successfully: ID {new_id}")                                                                           # Update position state with new SL info                               position['stop_loss'] = new_tsl_formatted # Update the main SL reference
                        position['sl_order_id'] = new_id                                       position['current_trailing_sl_price'] = new_tsl_formatted # Mark TSL active price
                        save_position_state() # Save the successful update                                                                                        else:                                                                      log_error(f"Failed to place new trailing SL order after cancelling old one {old_sl_id}. POSITION MAY BE UNPROTECTED.")                        position['sl_order_id'] = None # Mark SL as lost                                                                                              position['current_trailing_sl_price'] = None                           save_position_state()                                                                                                                 except Exception as place_e:                                                # Error logged by retry decorator or generic handler                                                                                          log_error(f"Error placing new trailing SL order: {place_e}. POSITION MAY BE UNPROTECTED.", exc_info=True)                                     position['sl_order_id'] = None                                         position['current_trailing_sl_price'] = None                           save_position_state()                                                                                                                                                                                       except ccxt.OrderNotFound:
                log_warning(f"Old SL order {old_sl_id} not found during TSL update (might have been filled/cancelled already).")                              # If SL filled, check_position_and_orders should handle the reset.                                                                            # Mark TSL as inactive for this cycle if SL ID is now gone.
                position['current_trailing_sl_price'] = None                           position['sl_order_id'] = None # Ensure SL ID is cleared if not found
                save_position_state() # Save the cleared state                     except Exception as cancel_e:                                              # Error logged by cancel_order_with_retry or its decorator
                log_error(f"Error cancelling old SL order {old_sl_id} during TSL update: {cancel_e}. Aborting TSL placement.", exc_info=False)
                # Do not proceed to place new SL if cancellation failed unexpectedly.                                                         
# --- Main Trading Loop ---                                            log_info(f"Initializing trading bot for {symbol} on {timeframe}...")   # Load position state ONCE at startup                                  load_position_state()                                                  log_info(f"Risk per trade: {risk_percentage*100:.2f}%")                log_info(f"Bot check interval: {sleep_interval_seconds} seconds ({sleep_interval_seconds/60:.1f} minutes)")
log_info(f"{NEON_YELLOW}Press Ctrl+C to stop the bot gracefully.{RESET}")                                                                     
while True:
    try:                                                                       cycle_start_time: pd.Timestamp = pd.Timestamp.now(tz='UTC')            print_cycle_divider(cycle_start_time)                                                                                                         # 1. Check Position/Order Consistency FIRST                            position_was_reset = check_position_and_orders(exchange, symbol)                                                                              display_position_status(position, price_precision_digits, amount_precision_digits) # Display status after check                               if position_was_reset:                                                      log_info("Position state reset by order check. Proceeding to check for new entries.")                                                         # Continue loop to immediately check for signals          
        # 2. Fetch Fresh OHLCV Data                                            ohlcv_df: Optional[pd.DataFrame] = fetch_ohlcv_data(exchange, symbol, timeframe, limit_count=data_limit)                                      if ohlcv_df is None or ohlcv_df.empty:                                     log_warning(f"Could not fetch valid OHLCV data. Waiting...")                                                                                  neon_sleep_timer(sleep_interval_seconds)                               continue                                                   
        # 3. Calculate Technical Indicators (Conditionally include ATR, Vol MA)                                                                       needs_atr = enable_atr_sl_tp or enable_trailing_stop                   needs_vol_ma = entry_volume_confirmation_enabled                       stoch_params = {'k': stoch_k, 'd': stoch_d, 'smooth_k': stoch_smooth_k}                                                                       df_with_indicators: Optional[pd.DataFrame] = calculate_technical_indicators(                                                                      ohlcv_df.copy(), # Use copy                                            rsi_len=rsi_length, stoch_params=stoch_params,                         calc_atr=needs_atr, atr_len=atr_length,                                calc_vol_ma=needs_vol_ma, vol_ma_len=entry_volume_ma_length
        )                                                                      if df_with_indicators is None or df_with_indicators.empty:                  log_warning(f"Indicator calculation failed. Waiting...")               neon_sleep_timer(sleep_interval_seconds)                               continue                                                                                                                                 # 4. Get Latest Data and Indicator Values                              if len(df_with_indicators) < 2:                                            log_warning("Not enough data points after indicator calculation. Waiting...")                                                                 neon_sleep_timer(sleep_interval_seconds)                               continue                                                                                                                                  latest_data: pd.Series = df_with_indicators.iloc[-1]                   # Construct indicator column names                                     rsi_col_name: str = f'RSI_{rsi_length}'                                stoch_k_col_name: str = f'STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}'                                                                        stoch_d_col_name: str = f'STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}'                                                                        # ATR column from pandas_ta is typically ATRr_LENGTH (raw ATR)         atr_col_name: str = f'ATRr_{atr_length}' if needs_atr else None        vol_ma_col_name: str = f'VOL_MA_{entry_volume_ma_length}' if needs_vol_ma else None                                                                                                                                  # Check required columns exist                                         required_base_cols: List[str] = ['close', 'high', 'low', 'open', 'volume']                                                                    required_indicator_cols: List[str] = [rsi_col_name, stoch_k_col_name, stoch_d_col_name]                                                       if needs_atr: required_indicator_cols.append(atr_col_name)             if needs_vol_ma and 'volume' in df_with_indicators.columns: required_indicator_cols.append(vol_ma_col_name)                           
        all_required_cols = required_base_cols + [col for col in required_indicator_cols if col is not None]                                          missing_cols = [col for col in all_required_cols if col not in df_with_indicators.columns]

        if missing_cols:                                                           log_error(f"Required columns missing in DataFrame: {missing_cols}. Check config/data. Available: {df_with_indicators.columns.tolist()}")                                                                             neon_sleep_timer(sleep_interval_seconds)                               continue                                                                                                                                  # Extract latest values safely, checking for NaNs                      try:                                                                       current_price: float = float(latest_data['close'])                     current_high: float = float(latest_data['high'])                       current_low: float = float(latest_data['low'])                         last_rsi: float = float(latest_data[rsi_col_name])                     last_stoch_k: float = float(latest_data[stoch_k_col_name])             last_stoch_d: float = float(latest_data[stoch_d_col_name])                                                                                    last_atr: Optional[float] = None                                       if needs_atr:                                                              atr_val = latest_data.get(atr_col_name)                                if pd.notna(atr_val):                                                      last_atr = float(atr_val)
                else:                                                                       log_warning(f"ATR value is NaN for the latest candle. ATR-based logic will be skipped.")                                                                                                                                                                                           current_volume: Optional[float] = None                                 last_volume_ma: Optional[float] = None                                 if needs_vol_ma and 'volume' in latest_data:                                vol_val = latest_data.get('volume')
                 vol_ma_val = latest_data.get(vol_ma_col_name)                          if pd.notna(vol_val): current_volume = float(vol_val)                  if pd.notna(vol_ma_val): last_volume_ma = float(vol_ma_val)                                                                                   if current_volume is None or last_volume_ma is None:                       log_warning("Volume or Volume MA is NaN. Volume confirmation may be skipped.")                                                                                                                              # Final check for essential NaNs                                       essential_values = [current_price, last_rsi, last_stoch_k, last_stoch_d]
            if any(pd.isna(v) for v in essential_values):                               raise ValueError("Essential indicator value is NaN.")                                                                                except (KeyError, ValueError, TypeError) as e:                              log_error(f"Error extracting latest data or NaN found: {e}. Data: {latest_data.to_dict()}", exc_info=True)                                    neon_sleep_timer(sleep_interval_seconds)                               continue                                                                                                                                 # Display Market Stats                                                 display_market_stats(current_price, last_rsi, last_stoch_k, last_stoch_d, last_atr, price_precision_digits)                                                                                                          # 5. Identify Order Blocks                                             bullish_ob, bearish_ob = identify_potential_order_block(                   df_with_indicators,                                                    vol_thresh_mult=ob_volume_threshold_multiplier,                        lookback_len=ob_lookback                                           )
        display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)                                                                                                                                                                                                                        # 6. Apply Trading Logic                                                                                                                      # --- Check if ALREADY IN A POSITION ---                               if position['status'] is not None:                                         log_info(f"Currently in {position['status'].upper()} position.")                                                                                                                                                     # A. Check Trailing Stop Logic (only if enabled and ATR available)                                                                            if enable_trailing_stop and last_atr is not None and last_atr > 0:                                                                                log_debug("Checking trailing stop condition...")
                update_trailing_stop(exchange, symbol, current_price, last_atr)                                                                               # Re-check position status in case TSL caused an exit (though check_pos should catch it next cycle)                                           # display_position_status(position, price_precision_digits, amount_precision_digits) # Display potential TSL update                       elif enable_trailing_stop and (last_atr is None or last_atr <= 0):                                                                                 log_warning("Trailing stop enabled but ATR is invalid. Skipping TSL check.")                                                                                                                                                                                                           # B. Check for Indicator-Based Exits (Optional Secondary Exit)                                                                                # Note: This logic is basic. More sophisticated exits could be added.
            # This acts as a fallback IF SL/TP somehow fail or if you want faster exits.                                                                  execute_indicator_exit = False                                         exit_reason = ""                                                       if position['status'] == 'long' and last_rsi > rsi_overbought:                                                                                     exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} > {rsi_overbought})"                                                                execute_indicator_exit = True                                     elif position['status'] == 'short' and last_rsi < rsi_oversold:                                                                                    exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} < {rsi_oversold})"                                                                  execute_indicator_exit = True                         
            if execute_indicator_exit:                                                 display_signal("Exit", position['status'], exit_reason)                log_warning(f"Attempting to close {position['status'].upper()} position via FALLBACK market order due to indicator signal.")
                                                                                       # IMPORTANT: Cancel existing SL and TP orders *before* sending market order                                                                   sl_id_to_cancel = position.get('sl_order_id')                          tp_id_to_cancel = position.get('tp_order_id')                          orders_cancelled_successfully = True # Assume success initially                                                                                                                                                      for order_id, order_type in [(sl_id_to_cancel, "SL"), (tp_id_to_cancel, "TP")]:                                                                   if order_id:                                                               log_info(f"Cancelling existing {order_type} order {order_id} before fallback exit...")                                                        try:                                                                       cancel_order_with_retry(exchange, order_id, symbol)
                        except ccxt.OrderNotFound:                                                 log_info(f"{order_type} order {order_id} already closed/cancelled.")                                                                      except Exception as e:                                                     log_error(f"Error cancelling {order_type} order {order_id} during fallback exit: {e}", exc_info=False)                                        orders_cancelled_successfully = False # Mark cancellation as potentially failed                                                                                                                          if not orders_cancelled_successfully:                                       log_error("Failed to cancel one or both SL/TP orders. Aborting fallback market exit to avoid potential issues. MANUAL INTERVENTION MAY BE REQUIRED.")                                                           else:                                                                      # Proceed with market exit order (use reduceOnly=True)                                                                                        close_side = 'sell' if position['status'] == 'long' else 'buy'                                                                                exit_qty = position['quantity']                                        if exit_qty is None or exit_qty <= 0:
                         log_error("Cannot place fallback exit order: Invalid quantity in position state.")                                                       else:                                                                      order_result = place_market_order(exchange, symbol, close_side, exit_qty, reduce_only=True)                                                                                                                          # Check if order likely filled (can be tricky with market orders)                                                                             # Status 'closed' is ideal, but some exchanges might return 'open' briefly                                                                    # We reset state assuming it worked, check_position_and_orders will confirm next cycle                                                        if order_result and order_result.get('id'):
                            log_info(f"Fallback {position['status']} position close order placed: ID {order_result.get('id', 'N/A')}")                                else:                                                                      log_error(f"Fallback market order placement FAILED for {position['status']} position.")                                                       # Critical: Bot tried to exit but failed. Manual intervention likely needed.                                                                                                                                     # Reset position state immediately after attempting market exit                                                                               # regardless of confirmation, as intent was to close.                                                                                         log_info("Resetting local position state after indicator-based market exit attempt.")                                                         position.update({                                                          'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,                                                                      'stop_loss': None, 'take_profit': None, 'entry_time': None,                                                                                   'sl_order_id': None, 'tp_order_id': None,                              'highest_price_since_entry': None, 'lowest_price_since_entry': None,                                                                          'current_trailing_sl_price': None                                  })                                                                     save_position_state()                                                  display_position_status(position, price_precision_digits, amount_precision_digits)                                                            # Exit loop for this cycle after attempting exit                                                                                              neon_sleep_timer(sleep_interval_seconds)                               continue                                                   else:                                                                       # If no indicator exit, just log monitoring status                     log_info(f"Monitoring {position['status'].upper()} position. Waiting for SL/TP ({position.get('sl_order_id')}/{position.get('tp_order_id')}) or TSL update.")                                                                                                                                                                                             # --- Check for NEW ENTRIES (only if not currently in a position) ---                                                                         else: # position['status'] is None                                         log_info("No active position. Checking for entry signals...")                                                                                                                                                        # --- Volume Confirmation Check ---                                    volume_confirmed = False                                               if entry_volume_confirmation_enabled:                                      if current_volume is not None and last_volume_ma is not None and last_volume_ma > 0:                                                              if current_volume > (last_volume_ma * entry_volume_multiplier):                                                                                   volume_confirmed = True                                                log_debug(f"Volume confirmed: Current Vol ({current_volume:.2f}) > MA Vol ({last_volume_ma:.2f}) * {entry_volume_multiplier}")                                                                                   else:                                                                      log_debug(f"Volume NOT confirmed: Current Vol ({current_volume:.2f}), MA Vol ({last_volume_ma:.2f}), Threshold ({last_volume_ma * entry_volume_multiplier:.2f})")                                            else:                                                                      log_debug("Volume confirmation enabled but volume or MA data is missing/invalid.")                                                    else:                                                                      volume_confirmed = True # Skip check if disabled
                                                                                   # --- Base Signal Conditions ---                                       base_long_signal = last_rsi < rsi_oversold and last_stoch_k < stoch_oversold                                                                  base_short_signal = last_rsi > rsi_overbought and last_stoch_k > stoch_overbought                                                                                                                                    # --- OB Price Check ---                                               long_ob_price_check = False                                            long_ob_reason_part = ""                                               if base_long_signal and bullish_ob:                                        ob_range = bullish_ob['high'] - bullish_ob['low']                      # Allow entry if price is within OB or up to 10% of OB range above high                                                                       entry_zone_high = bullish_ob['high'] + max(ob_range * 0.10, min_tick * 2) # Add min tick buffer                                               entry_zone_low = bullish_ob['low']                                     if entry_zone_low <= current_price <= entry_zone_high:                     long_ob_price_check = True                                             long_ob_reason_part = f"Price near Bullish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"                                                                      else:                                                                      log_debug(f"Base Long signal met, but price {current_price:.{price_precision_digits}f} outside Bullish OB entry zone.")               elif base_long_signal:                                                      log_debug("Base Long signal met, but no recent Bullish OB found.")                                                                                                                                              short_ob_price_check = False                                           short_ob_reason_part = ""                                              if base_short_signal and bearish_ob:                                       ob_range = bearish_ob['high'] - bearish_ob['low']
                # Allow entry if price is within OB or down to 10% of OB range below low
                entry_zone_low = bearish_ob['low'] - max(ob_range * 0.10, min_tick * 2)                                                                       entry_zone_high = bearish_ob['high']                                   if entry_zone_low <= current_price <= entry_zone_high:                     short_ob_price_check = True                                            short_ob_reason_part = f"Price near Bearish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                else:                                                                       log_debug(f"Base Short signal met, but price {current_price:.{price_precision_digits}f} outside Bearish OB entry zone.")             elif base_short_signal:                                                      log_debug("Base Short signal met, but no recent Bearish OB found.")                                                         
                                                                                   # --- Combine Conditions for Final Entry Signal ---                    long_entry_condition = base_long_signal and long_ob_price_check and volume_confirmed                                                          short_entry_condition = base_short_signal and short_ob_price_check and volume_confirmed                                                                                                                              long_reason = ""                                                       short_reason = ""                                                      if long_entry_condition:                                                    long_reason = (f"RSI ({last_rsi:.1f} < {rsi_oversold}), "                                                                                                    f"StochK ({last_stoch_k:.1f} < {stoch_oversold}), "                                                                                           f"{long_ob_reason_part}" +                                             (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))                                                      elif short_entry_condition:                                                 short_reason = (f"RSI ({last_rsi:.1f} > {rsi_overbought}), "                                                                                                  f"StochK ({last_stoch_k:.1f} > {stoch_overbought}), "                                                                                         f"{short_ob_reason_part}" +                                            (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))
            # Log reasons for failure if base signal was met but other checks failed                                                                      elif base_long_signal and long_ob_price_check and not volume_confirmed:                                                                            log_debug("Long OB/Price conditions met, but volume not confirmed.")                                                                     elif base_short_signal and short_ob_price_check and not volume_confirmed:                                                                          log_debug("Short OB/Price conditions met, but volume not confirmed.")                                                                                                                                                                                                                  # --- Execute Entry ---                                                if long_entry_condition:                                                   display_signal("Entry", "long", long_reason)                                                                                                  # Calculate SL/TP prices
                stop_loss_price = 0.0                                                  take_profit_price = 0.0                                                if enable_atr_sl_tp:                                                       if last_atr is None or last_atr <= 0:                                      log_error("Cannot calculate ATR SL/TP: Invalid ATR value. Skipping LONG entry.")                                                              continue # Skip to next cycle                                      stop_loss_price = current_price - (last_atr * atr_sl_multiplier)                                                                              take_profit_price = current_price + (last_atr * atr_tp_multiplier)                                                                            log_info(f"Calculated ATR-based SL: {stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP: {take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)")                     else: # Fixed percentage                                                   stop_loss_price = current_price * (1 - stop_loss_percentage)                                                                                  take_profit_price = current_price * (1 + take_profit_percentage)                                                                              log_info(f"Calculated Fixed % SL: {stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage*100:.1f}%), TP: {take_profit_price:.{price_precision_digits}f} ({take_profit_percentage*100:.1f}%)")                                                                                                                                                     # Adjust SL based on Bullish OB low if it provides tighter stop                                                                               # Ensure bullish_ob exists (should be true if long_entry_condition is met with OB check)                                                      if bullish_ob and bullish_ob['low'] > stop_loss_price:
                    # Set SL just below the OB low (add buffer)
                    adjusted_sl = bullish_ob['low'] * (1 - 0.0005) # Smaller buffer, e.g., 0.05%                                                                  # Ensure adjustment doesn't push SL too close or above current price
                    if adjusted_sl < current_price and adjusted_sl > stop_loss_price:                                                                                  stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))
                         log_info(f"Adjusted SL tighter based on Bullish OB low: {stop_loss_price:.{price_precision_digits}f}")                                   else:                                                                       log_warning(f"Could not adjust SL based on OB low {bullish_ob['low']:.{price_precision_digits}f} as it's too close or invalid.")                                                                                                                                                                                                                          # Final SL/TP validation                                               if stop_loss_price >= current_price or take_profit_price <= current_price:                                                                         log_error(f"Invalid SL/TP calculated: SL {stop_loss_price:.{price_precision_digits}f}, TP {take_profit_price:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping LONG entry.")                                                               continue                                                                                                                                 # Calculate position size                                              quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)                                         if quantity is None or quantity <= 0:                                      log_error("Failed to calculate valid position size. Skipping LONG entry.")                                                                else:                                                                      # Place entry market order                                             entry_order_result = place_market_order(exchange, symbol, 'buy', quantity, reduce_only=False) # Entry is not reduceOnly                       if entry_order_result and entry_order_result.get('status') == 'closed': # Check fill status                                                       entry_price_actual = entry_order_result.get('average', current_price) # Use filled price if available                                         filled_quantity = entry_order_result.get('filled', quantity) # Use filled qty if available                                                                                                                           log_info(f"Long position entry order filled: ID {entry_order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f} for {filled_quantity:.{amount_precision_digits}f}")                                                                                               # Place SL/TP orders *after* confirming entry                          sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'long', filled_quantity, stop_loss_price, take_profit_price)
                                                                                               # Update position state ONLY if entry was successful
                        position.update({                                                          'status': 'long',
                            'entry_price': entry_price_actual,                                     'quantity': filled_quantity,
                            'order_id': entry_order_result.get('id'),                              'stop_loss': stop_loss_price,                                          'take_profit': take_profit_price,
                            'entry_time': pd.Timestamp.now(tz='UTC'),                              'sl_order_id': sl_order.get('id') if sl_order else None,                                                                                      'tp_order_id': tp_order.get('id') if tp_order else None,
                            # Initialize TSL fields                                                'highest_price_since_entry': entry_price_actual,                                                                                              'lowest_price_since_entry': None,                                      'current_trailing_sl_price': None                                  })                                                                     save_position_state()                                                  display_position_status(position, price_precision_digits, amount_precision_digits) # Show new status                                                                                                                 if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):                                                                log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")                                                                                     else:                                                                      log_error("Failed to place or confirm fill for long entry order.")                                                                                                                                       elif short_entry_condition: # Use elif to prevent long/short in same cycle                                                                        display_signal("Entry", "short", short_reason)                                                                                                # Calculate SL/TP prices                                               stop_loss_price = 0.0                                                  take_profit_price = 0.0                                                if enable_atr_sl_tp:                                                       if last_atr is None or last_atr <= 0:                                      log_error("Cannot calculate ATR SL/TP: Invalid ATR value. Skipping SHORT entry.")
                        continue                                                           stop_loss_price = current_price + (last_atr * atr_sl_multiplier)                                                                              take_profit_price = current_price - (last_atr * atr_tp_multiplier)                                                                            log_info(f"Calculated ATR-based SL: {stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP: {take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)")                     else: # Fixed percentage
                    stop_loss_price = current_price * (1 + stop_loss_percentage)                                                                                  take_profit_price = current_price * (1 - take_profit_percentage)                                                                              log_info(f"Calculated Fixed % SL: {stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage*100:.1f}%), TP: {take_profit_price:.{price_precision_digits}f} ({take_profit_percentage*100:.1f}%)")                                                                                                                                                     # Adjust SL based on Bearish OB high if it provides tighter stop                                                                              if bearish_ob and bearish_ob['high'] < stop_loss_price:                     adjusted_sl = bearish_ob['high'] * (1 + 0.0005) # Small buffer above high                                                                     if adjusted_sl > current_price and adjusted_sl < stop_loss_price:
                         stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))                                                                     log_info(f"Adjusted SL tighter based on Bearish OB high: {stop_loss_price:.{price_precision_digits}f}")                                   else:                                                                      log_warning(f"Could not adjust SL based on OB high {bearish_ob['high']:.{price_precision_digits}f} as it's too close or invalid.")                                                                                                                                                 # Final SL/TP validation                                               if stop_loss_price <= current_price or take_profit_price >= current_price:                                                                         log_error(f"Invalid SL/TP calculated: SL {stop_loss_price:.{price_precision_digits}f}, TP {take_profit_price:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping SHORT entry.")                                                              continue                                          
                # Calculate position size
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)                                         if quantity is None or quantity <= 0:                                      log_error("Failed to calculate valid position size. Skipping SHORT entry.")                                                               else:                                                                      # Place entry market order                                             entry_order_result = place_market_order(exchange, symbol, 'sell', quantity, reduce_only=False)                                                if entry_order_result and entry_order_result.get('status') == 'closed':                                                                           entry_price_actual = entry_order_result.get('average', current_price)                                                                         filled_quantity = entry_order_result.get('filled', quantity)                                                          
                        log_info(f"Short position entry order filled: ID {entry_order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f} for {filled_quantity:.{amount_precision_digits}f}")                                                                      
                        # Place SL/TP orders                                                   sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'short', filled_quantity, stop_loss_price, take_profit_price)                                                                                                                                                                     # Update position state
                        position.update({
                            'status': 'short',                                                     'entry_price': entry_price_actual,                                     'quantity': filled_quantity,
                            'order_id': entry_order_result.get('id'),                              'stop_loss': stop_loss_price,                                          'take_profit': take_profit_price,
                            'entry_time': pd.Timestamp.now(tz='UTC'),                              'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None,                                                                                      # Initialize TSL fields
                            'highest_price_since_entry': None,
                            'lowest_price_since_entry': entry_price_actual,                                                                                               'current_trailing_sl_price': None                                  })                                                                     save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits)                                                                                                                                   if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):                                                                log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                    else:
                        log_error("Failed to place or confirm fill for short entry order.")                                                                                                                                      else: # No entry condition met                                             log_info("Conditions not met for new entry.")                                                                                 
        # 7. Wait for the next cycle                                           log_info(f"Cycle complete. Waiting for {sleep_interval_seconds} seconds...")                                                                  neon_sleep_timer(sleep_interval_seconds)                                                                                                  # --- Graceful Shutdown Handling ---                                   except KeyboardInterrupt:                                                  log_info("Keyboard interrupt detected (Ctrl+C). Stopping the bot...")                                                                         save_position_state()  # Save final state before exiting                                                                                      # Attempt to cancel open orders for the symbol                         if not SIMULATION_MODE:                                                    log_info(f"Attempting to cancel all open orders for {symbol}...")                                                                             try:                                                                       # Fetch open orders first (use retry)                                  @api_retry_decorator                                                   def fetch_open_orders_on_exit(exch, sym):
                    return exch.fetch_open_orders(sym)                                                                                                        open_orders = fetch_open_orders_on_exit(exchange, symbol)                                                                                                                                                            if not open_orders:                                                        log_info("No open orders found to cancel.")                        else:                                                                      log_warning(f"Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...")
                    cancelled_count = 0                                                    failed_count = 0                                                       for order in open_orders:                                                  order_id = order.get('id')                                             if not order_id: continue # Skip if no ID                              try:
                            log_info(f"Cancelling order ID: {order_id} (Type: {order.get('type', 'N/A')}, Side: {order.get('side','N/A')})...")                                                                                                  # Use cancel helper with retry                                         cancel_order_with_retry(exchange, order_id, symbol)                                                                                           cancelled_count += 1                                                   time.sleep(0.3) # Small delay between cancellations
                        except ccxt.OrderNotFound:                                                  log_info(f"Order {order_id} already closed/cancelled.")                                                                                       cancelled_count += 1 # Count as effectively cancelled                                                                                    except Exception as cancel_e:                                              # Error logged by helper                                               log_error(f"Failed to cancel order {order_id} on exit after retries.", exc_info=False)

Okay, I've integrated all the upgrade snippets into the provided base script (`trading_bot_neon_enhanced.py`), creating a complete, enhanced version.

This improved script incorporates:

1.  **ATR-Based Stop Loss and Take Profit (Optional):** Calculates ATR and uses it for SL/TP if enabled in the config. Includes OB adjustment logic.
2.  **Trailing Stop Loss (Optional):** Implements ATR-based trailing stop logic if enabled.
3.  **Configuration File (`config.json`):** Moves all parameters to an external JSON file for easier management.
4.  **Volume Confirmation (Optional):** Adds an optional volume check (Volume > MA Volume \* Multiplier) for entry signals.
5.  **Enhanced Error Handling (Retry Decorator):** Adds a robust retry mechanism with exponential backoff for common network/API errors using a decorator.

**First, create the `config.json` file in the same directory as the script:**

```json
# --- config.json ---
{
  "exchange_id": "bybit",
  "symbol": "BTC/USDT:USDT",
  "timeframe": "1h",
  "data_limit": 200,
  "sleep_interval_seconds": 900,

  "rsi_length": 14,
  "rsi_overbought": 70,
  "rsi_oversold": 30,

  "stoch_k": 14,
  "stoch_d": 3,
  "stoch_smooth_k": 3,
  "stoch_overbought": 80,
  "stoch_oversold": 20,

  "risk_percentage": 0.01,

  "enable_atr_sl_tp": true,
  "stop_loss_percentage": 0.02,       # Used only if enable_atr_sl_tp is false
  "take_profit_percentage": 0.04,     # Used only if enable_atr_sl_tp is false
  "atr_length": 14,                   # Used if enable_atr_sl_tp is true OR enable_trailing_stop is true
  "atr_sl_multiplier": 2.0,           # Used only if enable_atr_sl_tp is true
  "atr_tp_multiplier": 3.0,           # Used only if enable_atr_sl_tp is true

  "enable_trailing_stop": true,
  "trailing_stop_atr_multiplier": 1.5,      # How far behind the peak price to trail (in ATR units)
  "trailing_stop_activation_atr_multiplier": 1.0, # How much profit (in ATR units) needed to activate TSL

  "ob_volume_threshold_multiplier": 1.5,
  "ob_lookback": 10,                  # How many candles back to look for OB pattern & calc avg vol

  "entry_volume_confirmation_enabled": true,
  "entry_volume_ma_length": 20,
  "entry_volume_multiplier": 1.2,     # Volume must be > 1.2 * MA Volume

  "simulation_mode": true,            # IMPORTANT: Set to false for live trading!

  "retry_max_retries": 3,
  "retry_initial_delay": 5.0,
  "retry_backoff_factor": 2.0
}
```

**Now, here is the complete enhanced Python script (`trading_bot_neon_enhanced_v2.py`):**

```python
# trading_bot_neon_enhanced_v2.py
# Enhanced version incorporating ATR SL/TP, Trailing Stops, Config File,
# Volume Confirmation, and Retry Logic.

import ccxt
import os
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import pandas_ta as ta # Using pandas_ta for indicator calculations
import json # For config, pretty printing order details and saving state
import os.path # For checking if state file exists
from typing import Optional, Tuple, Dict, Any, List, Union
import functools # For retry decorator
import sys # For exit

# --- Colorama Initialization and Neon Palette ---
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True) # Autoreset ensures styles don't leak

    # Define neon color palette
    NEON_GREEN = Fore.GREEN + Style.BRIGHT
    NEON_PINK = Fore.MAGENTA + Style.BRIGHT
    NEON_CYAN = Fore.CYAN + Style.BRIGHT
    NEON_RED = Fore.RED + Style.BRIGHT
    NEON_YELLOW = Fore.YELLOW + Style.BRIGHT
    NEON_BLUE = Fore.BLUE + Style.BRIGHT
    RESET = Style.RESET_ALL # Although autoreset is on, good practice
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Warning: colorama not found. Neon styling will be disabled. Install with: pip install colorama")
    # Define dummy colors if colorama is not available
    NEON_GREEN = NEON_PINK = NEON_CYAN = NEON_RED = NEON_YELLOW = NEON_BLUE = RESET = ""
    COLORAMA_AVAILABLE = False

# --- Logging Configuration ---
log_format_base: str = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
# Configure root logger - handlers can be added later if needed (e.g., file handler)
logging.basicConfig(level=logging.INFO, format=log_format_base, datefmt='%Y-%m-%d %H:%M:%S')
logger: logging.Logger = logging.getLogger(__name__) # Get logger for this module

# --- Neon Display Functions ---

def print_neon_header():
    """Prints a neon-styled header banner."""
    print(f"{NEON_CYAN}{'=' * 70}{RESET}")
    print(f"{NEON_PINK}{Style.BRIGHT}     Enhanced RSI/OB Trader Neon Bot - Configurable v2     {RESET}")
    print(f"{NEON_CYAN}{'=' * 70}{RESET}")

def display_error_box(message: str):
    """Displays an error message in a neon box."""
    box_width = 70
    print(f"{NEON_RED}{'!' * box_width}{RESET}")
    print(f"{NEON_RED}! {message:^{box_width-4}} !{RESET}")
    print(f"{NEON_RED}{'!' * box_width}{RESET}")

def display_warning_box(message: str):
    """Displays a warning message in a neon box."""
    box_width = 70
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")
    print(f"{NEON_YELLOW}~ {message:^{box_width-4}} ~{RESET}")
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")

# Custom logging wrappers with colorama
def log_info(msg: str):
    logger.info(f"{NEON_GREEN}{msg}{RESET}")

def log_error(msg: str, exc_info=False):
    # Show first line in box for prominence, log full message
    first_line = msg.split('\n', 1)[0]
    display_error_box(first_line)
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)

def log_warning(msg: str):
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")

def log_debug(msg: str):
     # Use a less prominent color for debug
    logger.debug(f"{Fore.WHITE}{msg}{RESET}") # Simple white for debug

def print_cycle_divider(timestamp: pd.Timestamp):
    """Prints a neon divider for each trading cycle."""
    box_width = 70
    print(f"\n{NEON_BLUE}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}Cycle Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{NEON_BLUE}{'=' * box_width}{RESET}")

def display_position_status(position: Dict[str, Any], price_precision: int = 4, amount_precision: int = 8):
    """Displays position status with neon colors and formatted values."""
    status = position.get('status', None)
    entry_price = position.get('entry_price')
    quantity = position.get('quantity')
    sl = position.get('stop_loss')
    tp = position.get('take_profit')
    tsl = position.get('current_trailing_sl_price') # Added TSL display

    entry_str = f"{entry_price:.{price_precision}f}" if isinstance(entry_price, (float, int)) else "N/A"
    qty_str = f"{quantity:.{amount_precision}f}" if isinstance(quantity, (float, int)) else "N/A"
    sl_str = f"{sl:.{price_precision}f}" if isinstance(sl, (float, int)) else "N/A"
    tp_str = f"{tp:.{price_precision}f}" if isinstance(tp, (float, int)) else "N/A"
    tsl_str = f" | TSL: {tsl:.{price_precision}f}" if isinstance(tsl, (float, int)) else "" # Show TSL only if active

    if status == 'long':
        color = NEON_GREEN
        status_text = "LONG"
    elif status == 'short':
        color = NEON_RED
        status_text = "SHORT"
    else:
        color = NEON_CYAN
        status_text = "None"

    print(f"{color}Position Status: {status_text}{RESET} | Entry: {entry_str} | Qty: {qty_str} | SL: {sl_str} | TP: {tp_str}{tsl_str}")


def display_market_stats(current_price: float, rsi: float, stoch_k: float, stoch_d: float, atr: Optional[float], price_precision: int):
    """Displays market stats in a neon-styled panel."""
    print(f"{NEON_PINK}--- Market Stats ---{RESET}")
    print(f"{NEON_GREEN}Price:{RESET}  {current_price:.{price_precision}f}")
    print(f"{NEON_CYAN}RSI:{RESET}    {rsi:.2f}")
    print(f"{NEON_YELLOW}StochK:{RESET} {stoch_k:.2f}")
    print(f"{NEON_YELLOW}StochD:{RESET} {stoch_d:.2f}")
    # Use price_precision for ATR display as it's a price volatility measure
    if atr is not None:
         print(f"{NEON_BLUE}ATR:{RESET}    {atr:.{price_precision}f}") # Display ATR
    print(f"{NEON_PINK}--------------------{RESET}")

def display_order_blocks(bullish_ob: Optional[Dict], bearish_ob: Optional[Dict], price_precision: int):
    """Displays order blocks with neon colors."""
    found = False
    if bullish_ob:
        print(f"{NEON_GREEN}Bullish OB:{RESET} {bullish_ob['time'].strftime('%H:%M')} | Low: {bullish_ob['low']:.{price_precision}f} | High: {bullish_ob['high']:.{price_precision}f}")
        found = True
    if bearish_ob:
        print(f"{NEON_RED}Bearish OB:{RESET} {bearish_ob['time'].strftime('%H:%M')} | Low: {bearish_ob['low']:.{price_precision}f} | High: {bearish_ob['high']:.{price_precision}f}")
        found = True
    if not found:
        print(f"{NEON_BLUE}Order Blocks: None detected in recent data.{RESET}")


def display_signal(signal_type: str, direction: str, reason: str):
    """Displays trading signals with neon colors."""
    if direction.lower() == 'long':
        color = NEON_GREEN
    elif direction.lower() == 'short':
        color = NEON_RED
    else:
        color = NEON_YELLOW # For general signals/alerts

    print(f"{color}{Style.BRIGHT}*** {signal_type.upper()} {direction.upper()} SIGNAL ***{RESET}\n   Reason: {reason}")

def neon_sleep_timer(seconds: int):
    """Displays a neon countdown timer."""
    if not COLORAMA_AVAILABLE or seconds <= 0: # Fallback if colorama not installed or zero sleep
        if seconds > 0:
            print(f"Sleeping for {seconds} seconds...")
            time.sleep(seconds)
        return

    interval = 0.5 # Update interval for the timer display
    steps = int(seconds / interval)
    for i in range(steps, -1, -1):
        remaining_seconds = max(0, int(i * interval)) # Ensure non-negative
        # Flashing effect for last 5 seconds
        color = NEON_RED if remaining_seconds <= 5 and i % 2 == 0 else NEON_YELLOW
        print(f"{color}Next cycle in: {remaining_seconds} seconds... {Style.RESET_ALL}", end='\r')
        time.sleep(interval)
    print(" " * 50, end='\r')  # Clear line after countdown

def print_shutdown_message():
    """Prints a neon shutdown message."""
    box_width = 70
    print(f"\n{NEON_PINK}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}{Style.BRIGHT}{'RSI/OB Trader Bot Stopped - Goodbye!':^{box_width}}{RESET}")
    print(f"{NEON_PINK}{'=' * box_width}{RESET}")


# --- Constants ---
DEFAULT_PRICE_PRECISION: int = 4
DEFAULT_AMOUNT_PRECISION: int = 8
POSITION_STATE_FILE = 'position_state.json' # Define filename for state persistence
CONFIG_FILE = 'config.json'

# --- Retry Decorator (From Snippet 5) ---
RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout
)

def retry_api_call(max_retries: int = 3, initial_delay: float = 5.0, backoff_factor: float = 2.0):
    """Decorator factory to create retry decorators with specific parameters."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    retries += 1
                    if retries >= max_retries:
                        log_error(f"API call '{func.__name__}' failed after {max_retries} retries. Last error: {e}", exc_info=False)
                        raise # Re-raise the last exception
                    else:
                        log_warning(f"API call '{func.__name__}' failed with {type(e).__name__}. Retrying in {delay:.1f}s... (Attempt {retries}/{max_retries})")
                        # Use neon sleep if available, otherwise time.sleep
                        try:
                            neon_sleep_timer(int(delay))
                        except NameError:
                            time.sleep(delay)
                        delay *= backoff_factor # Exponential backoff
                except Exception as e:
                     # Handle non-retryable exceptions immediately
                     log_error(f"Non-retryable error in API call '{func.__name__}': {e}", exc_info=True)
                     raise # Re-raise immediately
            # This line should theoretically not be reached if max_retries > 0
            # but added for safety to ensure function returns or raises.
            raise RuntimeError(f"API call '{func.__name__}' failed unexpectedly after retries.")
        return wrapper
    return decorator


# --- Configuration Loading (From Snippet 3) ---
def load_config(filename: str = CONFIG_FILE) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    log_info(f"Attempting to load configuration from '{filename}'...")
    try:
        with open(filename, 'r') as f:
            config_data = json.load(f)
        log_info(f"Configuration loaded successfully from {filename}")
        # Basic validation (presence of essential keys)
        required_keys = ["exchange_id", "symbol", "timeframe", "risk_percentage", "simulation_mode"]
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
             log_error(f"CRITICAL: Missing required configuration keys in '{filename}': {missing_keys}")
             sys.exit(1) # Use sys.exit for clean exit
        # TODO: Add more specific validation (types, ranges) if needed
        return config_data
    except FileNotFoundError:
        log_error(f"CRITICAL: Configuration file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_error(f"CRITICAL: Error decoding JSON from configuration file '{filename}': {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"CRITICAL: An unexpected error occurred loading configuration: {e}", exc_info=True)
        sys.exit(1)

# Load config early
config = load_config()

# Create the decorator instance using config values
api_retry_decorator = retry_api_call(
    max_retries=config.get("retry_max_retries", 3),
    initial_delay=config.get("retry_initial_delay", 5.0),
    backoff_factor=config.get("retry_backoff_factor", 2.0)
)

# --- Environment & Exchange Setup ---
print_neon_header() # Show header early
load_dotenv()
log_info("Attempting to load environment variables from .env file (for API keys)...")

exchange_id: str = config.get("exchange_id", "bybit").lower()
api_key_env_var: str = f"{exchange_id.upper()}_API_KEY"
secret_key_env_var: str = f"{exchange_id.upper()}_SECRET_KEY"
passphrase_env_var: str = f"{exchange_id.upper()}_PASSPHRASE" # Some exchanges like kucoin, okx use this

api_key: Optional[str] = os.getenv(api_key_env_var)
secret: Optional[str] = os.getenv(secret_key_env_var)
passphrase: str = os.getenv(passphrase_env_var, '')

if not api_key or not secret:
    log_error(f"CRITICAL: API Key or Secret not found using env vars '{api_key_env_var}' and '{secret_key_env_var}'. "
              f"Please ensure these exist in your .env file or environment.")
    sys.exit(1)

log_info(f"Attempting to connect to exchange: {exchange_id}")
exchange: ccxt.Exchange
try:
    exchange_class = getattr(ccxt, exchange_id)
    exchange_config: Dict[str, Any] = {
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True, # CCXT internal rate limiting
        'options': {
            # Adjust 'swap' or 'spot' based on your market type in config/symbol
            'defaultType': 'swap' if ':' in config.get("symbol", "") else 'spot',
            # 'adjustForTimeDifference': True, # Optional: Helps with timestamp issues
        }
    }
    if passphrase:
        log_info("Passphrase detected, adding to exchange configuration.")
        exchange_config['password'] = passphrase

    exchange = exchange_class(exchange_config)

    # Load markets with retry mechanism applied
    @api_retry_decorator
    def load_markets_with_retry(exch_instance):
         log_info("Loading markets...")
         exch_instance.load_markets()

    load_markets_with_retry(exchange)

    log_info(f"Successfully connected to {exchange_id}. Markets loaded ({len(exchange.markets)} symbols found).")

except ccxt.AuthenticationError as e:
    log_error(f"Authentication failed connecting to {exchange_id}. Check API Key/Secret/Passphrase. Error: {e}")
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    log_error(f"Exchange {exchange_id} is not available. Error: {e}")
    sys.exit(1)
except AttributeError:
    log_error(f"Exchange ID '{exchange_id}' not found in ccxt library.")
    sys.exit(1)
except Exception as e:
    # Check if it's a retryable error that failed all retries (already logged by decorator)
    if not isinstance(e, RETRYABLE_EXCEPTIONS):
        log_error(f"An unexpected error occurred during exchange initialization or market loading: {e}", exc_info=True)
    sys.exit(1)


# --- Trading Parameters (Loaded from config) ---
symbol: str = config.get("symbol", "").strip().upper()
if not symbol:
    log_error("CRITICAL: 'symbol' not specified in config.json")
    sys.exit(1)

# Validate symbol against exchange markets and get precision info
try:
    if symbol not in exchange.markets:
        log_warning(f"Symbol '{symbol}' from config not found or not supported on {exchange_id}.")
        available_symbols: List[str] = list(exchange.markets.keys())
        # Filter for potential swap markets if needed
        if exchange_config.get('options', {}).get('defaultType') == 'swap':
             available_symbols = [s for s in available_symbols if ':' in s or exchange.markets[s].get('swap')]
        log_info(f"Some available symbols ({len(available_symbols)} total): {available_symbols[:15]}...")
        sys.exit(1)

    log_info(f"Using trading symbol from config: {symbol}")
    # Store precision info globally for convenience
    market_info = exchange.markets[symbol]
    price_precision_digits: int = int(market_info.get('precision', {}).get('price', DEFAULT_PRICE_PRECISION))
    amount_precision_digits: int = int(market_info.get('precision', {}).get('amount', DEFAULT_AMOUNT_PRECISION))
    min_tick: float = 1 / (10 ** price_precision_digits) if price_precision_digits > 0 else 0.01 # Smallest price change

    log_info(f"Symbol Precision | Price: {price_precision_digits} decimals (Min Tick: {min_tick}), Amount: {amount_precision_digits} decimals")

except Exception as e:
    log_error(f"An error occurred while validating the symbol or getting precision: {e}", exc_info=True)
    sys.exit(1)

timeframe: str = config.get("timeframe", "1h")
rsi_length: int = int(config.get("rsi_length", 14))
rsi_overbought: int = int(config.get("rsi_overbought", 70))
rsi_oversold: int = int(config.get("rsi_oversold", 30))
stoch_k: int = int(config.get("stoch_k", 14))
stoch_d: int = int(config.get("stoch_d", 3))
stoch_smooth_k: int = int(config.get("stoch_smooth_k", 3))
stoch_overbought: int = int(config.get("stoch_overbought", 80))
stoch_oversold: int = int(config.get("stoch_oversold", 20))
data_limit: int = int(config.get("data_limit", 200))
sleep_interval_seconds: int = int(config.get("sleep_interval_seconds", 900))
risk_percentage: float = float(config.get("risk_percentage", 0.01))

# SL/TP and Trailing Stop Parameters (Conditional loading)
enable_atr_sl_tp: bool = config.get("enable_atr_sl_tp", False)
enable_trailing_stop: bool = config.get("enable_trailing_stop", False)
atr_length: int = int(config.get("atr_length", 14)) # Needed if either ATR SL/TP or TSL is enabled

# Validate ATR length if needed
needs_atr = enable_atr_sl_tp or enable_trailing_stop
if needs_atr and atr_length <= 0:
    log_error(f"CRITICAL: 'atr_length' must be positive ({atr_length}) if ATR SL/TP or Trailing Stop is enabled.")
    sys.exit(1)

if enable_atr_sl_tp:
    atr_sl_multiplier: float = float(config.get("atr_sl_multiplier", 2.0))
    atr_tp_multiplier: float = float(config.get("atr_tp_multiplier", 3.0))
    log_info(f"Using ATR-based Stop Loss ({atr_sl_multiplier}x ATR) and Take Profit ({atr_tp_multiplier}x ATR).")
    if atr_sl_multiplier <= 0 or atr_tp_multiplier <= 0:
         log_error("CRITICAL: ATR SL/TP multipliers must be positive.")
         sys.exit(1)
else:
    stop_loss_percentage: float = float(config.get("stop_loss_percentage", 0.02))
    take_profit_percentage: float = float(config.get("take_profit_percentage", 0.04))
    log_info(f"Using Fixed Percentage Stop Loss ({stop_loss_percentage*100:.1f}%) and Take Profit ({take_profit_percentage*100:.1f}%).")
    if stop_loss_percentage <= 0 or take_profit_percentage <= 0:
         log_error("CRITICAL: Fixed SL/TP percentages must be positive.")
         sys.exit(1)

if enable_trailing_stop:
    trailing_stop_atr_multiplier: float = float(config.get("trailing_stop_atr_multiplier", 1.5))
    trailing_stop_activation_atr_multiplier: float = float(config.get("trailing_stop_activation_atr_multiplier", 1.0))
    log_info(f"Trailing Stop Loss is ENABLED (Activate @ {trailing_stop_activation_atr_multiplier}x ATR profit, Trail @ {trailing_stop_atr_multiplier}x ATR).")
    if trailing_stop_atr_multiplier <= 0 or trailing_stop_activation_atr_multiplier < 0: # Activation can be 0
         log_error("CRITICAL: Trailing stop ATR multiplier must be positive, activation multiplier non-negative.")
         sys.exit(1)
else:
    log_info("Trailing Stop Loss is DISABLED.")

# OB Parameters
ob_volume_threshold_multiplier: float = float(config.get("ob_volume_threshold_multiplier", 1.5))
ob_lookback: int = int(config.get("ob_lookback", 10))
if ob_lookback <= 0:
    log_error("CRITICAL: 'ob_lookback' must be positive.")
    sys.exit(1)

# Volume Confirmation Parameters
entry_volume_confirmation_enabled: bool = config.get("entry_volume_confirmation_enabled", True)
entry_volume_ma_length: int = int(config.get("entry_volume_ma_length", 20))
entry_volume_multiplier: float = float(config.get("entry_volume_multiplier", 1.2))
if entry_volume_confirmation_enabled:
    log_info(f"Entry Volume Confirmation: ENABLED (Vol > {entry_volume_multiplier}x MA({entry_volume_ma_length}))")
    if entry_volume_ma_length <= 0 or entry_volume_multiplier <= 0:
         log_error("CRITICAL: Volume MA length and multiplier must be positive.")
         sys.exit(1)
else:
    log_info("Entry Volume Confirmation: DISABLED")

# --- Simulation Mode ---
SIMULATION_MODE = config.get("simulation_mode", True) # Default to TRUE for safety
if SIMULATION_MODE:
    log_warning("SIMULATION MODE IS ACTIVE (set in config.json). No real orders will be placed.")
else:
    log_warning("!!! LIVE TRADING MODE IS ACTIVE (set in config.json). REAL ORDERS WILL BE PLACED. !!!")
    # Add confirmation step for live trading
    try:
        user_confirm = input(f"{NEON_RED}TYPE 'LIVE' TO CONFIRM LIVE TRADING or press Enter to exit: {RESET}")
        if user_confirm.strip().upper() != "LIVE":
            log_info("Live trading not confirmed. Exiting.")
            sys.exit(0)
        log_info("Live trading confirmed by user.")
    except EOFError: # Handle non-interactive environments
         log_error("Cannot get user confirmation in non-interactive mode. Exiting live mode.")
         sys.exit(1)


# --- Position Management State (Includes Trailing Stop fields) ---
position: Dict[str, Any] = {
    'status': None,         # None, 'long', or 'short'
    'entry_price': None,    # Float
    'quantity': None,       # Float
    'order_id': None,       # String (ID of the entry order)
    'stop_loss': None,      # Float (Initial or last manually set SL price)
    'take_profit': None,    # Float (Price level)
    'entry_time': None,     # pd.Timestamp
    'sl_order_id': None,    # String (ID of the open SL order)
    'tp_order_id': None,    # String (ID of the open TP order)
    # Fields for Trailing Stop Loss
    'highest_price_since_entry': None, # For long positions
    'lowest_price_since_entry': None,  # For short positions
    'current_trailing_sl_price': None # Active trailing SL price (can differ from 'stop_loss' if TSL active)
}

# --- State Saving and Resumption Functions ---
def save_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Saves position state to a JSON file."""
    global position
    try:
        # Create a copy to serialize Timestamp safely
        state_to_save = position.copy()
        if isinstance(state_to_save.get('entry_time'), pd.Timestamp):
            # Ensure timezone information is handled correctly (ISO format preserves it)
            state_to_save['entry_time'] = state_to_save['entry_time'].isoformat()

        with open(filename, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        log_debug(f"Position state saved to {filename}") # Use debug for less noise
    except Exception as e:
        log_error(f"Error saving position state to {filename}: {e}", exc_info=True)

def load_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Loads position state from a JSON file, handling potential missing keys."""
    global position
    log_info(f"Attempting to load position state from '{filename}'...")
    # Use the current global `position` dict as the template for keys and defaults
    default_position_state = position.copy()

    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                loaded_state: Dict[str, Any] = json.load(f)

            # Convert entry_time back to Timestamp if present and not None
            entry_time_str = loaded_state.get('entry_time')
            if entry_time_str:
                try:
                    # Attempt parsing, assume ISO format which includes timezone if saved correctly
                    ts = pd.Timestamp(entry_time_str)
                    # Ensure it's timezone-aware (UTC is preferred)
                    if ts.tzinfo is None:
                       ts = ts.tz_localize('UTC')
                    else:
                       ts = ts.tz_convert('UTC')
                    loaded_state['entry_time'] = ts
                except ValueError:
                    log_error(f"Could not parse entry_time '{entry_time_str}' from state file. Setting to None.")
                    loaded_state['entry_time'] = None

            # Update only the keys that exist in the current `position` structure
            updated_count = 0
            missing_in_file = []
            extra_in_file = []

            current_keys = set(default_position_state.keys())
            loaded_keys = set(loaded_state.keys())

            for key in current_keys:
                 if key in loaded_keys:
                     # Basic type check (optional but recommended)
                     # if isinstance(loaded_state[key], type(default_position_state[key])) or default_position_state[key] is None:
                     position[key] = loaded_state[key]
                     updated_count += 1
                     # else:
                     #    log_warning(f"Type mismatch for key '{key}' in state file. Expected {type(default_position_state[key])}, got {type(loaded_state[key])}. Using default.")
                     #    position[key] = default_position_state[key] # Reset to default on type mismatch
                 else:
                     missing_in_file.append(key)
                     position[key] = default_position_state[key] # Reset to default if missing in file

            extra_in_file = list(loaded_keys - current_keys)
            if 'entry_time' in extra_in_file and position['entry_time'] is not None: # Ignore entry_time if parsed
                extra_in_file.remove('entry_time')


            if missing_in_file:
                 log_warning(f"Keys missing in state file '{filename}' (reset to default): {missing_in_file}")
            if extra_in_file:
                 log_warning(f"Extra keys found in state file '{filename}' (ignored): {extra_in_file}")


            log_info(f"Position state loaded from {filename}. Updated {updated_count} fields.")
            display_position_status(position, price_precision_digits, amount_precision_digits) # Display loaded state
        else:
            log_info(f"No position state file found at {filename}. Starting with default state.")
    except json.JSONDecodeError as e:
        log_error(f"Error decoding JSON from state file {filename}: {e}. Starting with default state.", exc_info=True)
        position = default_position_state # Reset to default
    except Exception as e:
        log_error(f"Error loading position state from {filename}: {e}. Starting with default state.", exc_info=True)
        position = default_position_state # Reset to default


# --- Data Fetching Function (Decorated) ---
@api_retry_decorator
def fetch_ohlcv_data(exchange_instance: ccxt.Exchange, trading_symbol: str, tf: str, limit_count: int) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data and returns it as a pandas DataFrame. Retries on network errors."""
    log_debug(f"Fetching {limit_count} candles for {trading_symbol} on {tf} timeframe...")
    try:
        # Check if the market exists locally before fetching
        if trading_symbol not in exchange_instance.markets:
             log_error(f"Symbol {trading_symbol} not loaded in exchange markets.")
             return None

        ohlcv: List[list] = exchange_instance.fetch_ohlcv(trading_symbol, tf, limit=limit_count)
        if not ohlcv:
            log_warning(f"No OHLCV data returned for {trading_symbol} ({tf}). Exchange might be down or symbol inactive.")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        numeric_cols: List[str] = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # Use errors='coerce' to turn invalid parsing into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_rows: int = len(df)
        # Drop rows where any of the essential numeric columns are NaN
        df.dropna(subset=numeric_cols, inplace=True)
        rows_dropped: int = initial_rows - len(df)
        if rows_dropped > 0:
            log_debug(f"Dropped {rows_dropped} rows with invalid OHLCV data.")

        if df.empty:
            log_warning(f"DataFrame became empty after cleaning for {trading_symbol} ({tf}).")
            return None

        log_debug(f"Successfully fetched and processed {len(df)} candles for {trading_symbol}.")
        return df

    # Keep handling for non-retryable CCXT errors and general exceptions
    except ccxt.BadSymbol as e: # Non-retryable symbol issue
         log_error(f"BadSymbol error fetching OHLCV for {trading_symbol}: {e}. Check symbol format/availability.")
         return None
    except ccxt.ExchangeError as e: # Other exchange-specific errors
        log_error(f"Exchange specific error fetching OHLCV for {trading_symbol}: {e}")
        return None
    except Exception as e:
        # Error already logged by retry decorator if it was a retryable error that failed
        # Log here only if it's an unexpected non-CCXT error within this function's logic
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"An unexpected non-retryable error occurred fetching OHLCV: {e}", exc_info=True)
        return None # Return None if any exception occurred


# --- Enhanced Order Block Identification Function ---
def identify_potential_order_block(df: pd.DataFrame, vol_thresh_mult: float, lookback_len: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Identifies the most recent potential bullish and bearish order blocks based on
    a bearish/bullish candle followed by a high-volume reversal candle that sweeps liquidity.
    Uses config parameters for volume threshold multiplier and lookback length.
    """
    if df is None or df.empty or 'volume' not in df.columns or len(df) < lookback_len + 2:
        log_warning(f"Not enough data or missing volume column for OB detection (need > {lookback_len + 1} rows with volume, got {len(df)}).")
        return None, None

    bullish_ob: Optional[Dict[str, Any]] = None
    bearish_ob: Optional[Dict[str, Any]] = None

    try:
        # Calculate average volume excluding the last (potentially incomplete) candle
        completed_candles_df = df.iloc[:-1]
        avg_volume = 0
        # Ensure enough completed candles for rolling mean, use min_periods for robustness
        if len(completed_candles_df) >= 1:
             avg_volume = completed_candles_df['volume'].rolling(window=lookback_len, min_periods=max(1, lookback_len // 2)).mean().iloc[-1]
             if pd.isna(avg_volume): # Handle case where mean is NaN (e.g., all volumes were NaN)
                 avg_volume = 0

        volume_threshold = avg_volume * vol_thresh_mult if avg_volume > 0 else float('inf')
        log_debug(f"OB Analysis | Lookback: {lookback_len}, Avg Vol: {avg_volume:.2f}, Threshold Vol: {volume_threshold:.2f}")

        # Iterate backwards from second-to-last candle to find the most recent OBs
        # Look back up to 'lookback_len' candles for the pattern start
        search_end_index = max(-1, len(df) - lookback_len - 2)
        for i in range(len(df) - 2, search_end_index, -1):
            if i < 1: break # Need prev_candle

            candle = df.iloc[i]         # Potential "trigger" or "reversal" candle
            prev_candle = df.iloc[i-1]  # Potential candle forming the OB zone

            # Check for NaN values in the candles being examined
            if candle.isnull().any() or prev_candle.isnull().any():
                log_debug(f"Skipping OB check at index {i} due to NaN values.")
                continue

            is_high_volume = candle['volume'] > volume_threshold
            is_bullish_reversal = candle['close'] > candle['open']
            is_bearish_reversal = candle['close'] < candle['open']
            prev_is_bearish = prev_candle['close'] < prev_candle['open']
            prev_is_bullish = prev_candle['close'] > prev_candle['open']

            # --- Potential Bullish OB ---
            # Bearish prev_candle + High-volume Bullish reversal sweeping prev_low + closing strong
            if (not bullish_ob and # Find most recent
                prev_is_bearish and is_bullish_reversal and is_high_volume and
                candle['low'] < prev_candle['low'] and candle['close'] > prev_candle['high']): # Sweep and strong close
                bullish_ob = {
                    'high': prev_candle['high'], 'low': prev_candle['low'],
                    'time': prev_candle.name, # Timestamp of the bearish candle
                    'type': 'bullish'
                }
                log_debug(f"Potential Bullish OB found at {prev_candle.name.strftime('%Y-%m-%d %H:%M')} (Trigger: {candle.name.strftime('%H:%M')})")
                if bearish_ob: break # Found both most recent

            # --- Potential Bearish OB ---
            # Bullish prev_candle + High-volume Bearish reversal sweeping prev_high + closing strong
            elif (not bearish_ob and # Find most recent (use elif as one pair can't be both)
                  prev_is_bullish and is_bearish_reversal and is_high_volume and
                  candle['high'] > prev_candle['high'] and candle['close'] < prev_candle['low']): # Sweep and strong close
                bearish_ob = {
                    'high': prev_candle['high'], 'low': prev_candle['low'],
                    'time': prev_candle.name, # Timestamp of the bullish candle
                    'type': 'bearish'
                }
                log_debug(f"Potential Bearish OB found at {prev_candle.name.strftime('%Y-%m-%d %H:%M')} (Trigger: {candle.name.strftime('%H:%M')})")
                if bullish_ob: break # Found both most recent

            # Optimization: If we've found both types, no need to look further back
            if bullish_ob and bearish_ob:
                break

        return bullish_ob, bearish_ob

    except Exception as e:
        log_error(f"Error in order block identification: {e}", exc_info=True)
        return None, None


# --- Indicator Calculation Function (Adds ATR and Volume MA conditionally) ---
def calculate_technical_indicators(df: Optional[pd.DataFrame],
                                  rsi_len: int, stoch_params: Dict, # Pass params explicitly
                                  calc_atr: bool = False, atr_len: int = 14,
                                  calc_vol_ma: bool = False, vol_ma_len: int = 20
                                  ) -> Optional[pd.DataFrame]:
    """Calculates technical indicators (RSI, Stoch, optional ATR, optional Vol MA)."""
    if df is None or df.empty:
        log_warning("Input DataFrame is None or empty for indicator calculation.")
        return None

    log_debug(f"Calculating indicators on DataFrame with {len(df)} rows...")
    original_columns = set(df.columns)
    calculated_indicators = []
    try:
        # Calculate RSI
        df.ta.rsi(length=rsi_len, append=True)
        calculated_indicators.append(f'RSI_{rsi_len}')

        # Calculate Stochastic Oscillator
        df.ta.stoch(k=stoch_params['k'], d=stoch_params['d'], smooth_k=stoch_params['smooth_k'], append=True)
        calculated_indicators.append(f'STOCHk_{stoch_params["k"]}_{stoch_params["d"]}_{stoch_params["smooth_k"]}')
        calculated_indicators.append(f'STOCHd_{stoch_params["k"]}_{stoch_params["d"]}_{stoch_params["smooth_k"]}')


        # Calculate ATR if needed (for ATR SL/TP or Trailing Stop)
        # pandas_ta typically names the raw ATR column 'ATRr_LENGTH'
        atr_col_name_expected = f'ATRr_{atr_len}'
        if calc_atr:
            df.ta.atr(length=atr_len, append=True)
            log_debug(f"ATR ({atr_len}) calculated.")
            calculated_indicators.append(atr_col_name_expected)

        # Calculate Volume MA if needed (for Volume Confirmation)
        vol_ma_col_name_expected = f'VOL_MA_{vol_ma_len}'
        if calc_vol_ma:
            if 'volume' in df.columns:
                # Use min_periods to handle start of series
                df[vol_ma_col_name_expected] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
                log_debug(f"Volume MA ({vol_ma_len}) calculated.")
                calculated_indicators.append(vol_ma_col_name_expected)
            else:
                log_warning("Volume column not found, cannot calculate Volume MA.")

        new_columns: List[str] = list(set(df.columns) - original_columns)
        # Verify that expected columns were actually added
        missing_calc = [ind for ind in calculated_indicators if ind not in new_columns and ind in df.columns] # Check if calculation failed silently
        if missing_calc:
             log_warning(f"Some indicators might not have been calculated properly: {missing_calc}")

        log_debug(f"Indicators calculated. Columns added: {sorted(new_columns)}")

        initial_len: int = len(df)
        # Drop rows with NaN values generated by indicators (only check calculated indicator columns)
        indicator_cols_present = [col for col in calculated_indicators if col in df.columns]
        if indicator_cols_present:
            df.dropna(subset=indicator_cols_present, inplace=True)
        else:
             log_warning("No indicator columns found to drop NaNs from.")

        rows_dropped_nan: int = initial_len - len(df)
        if rows_dropped_nan > 0:
             log_debug(f"Dropped {rows_dropped_nan} rows with NaN values after indicator calculation.")

        if df.empty:
            log_warning("DataFrame became empty after dropping NaN rows from indicators.")
            return None

        log_debug(f"Indicator calculation complete. DataFrame now has {len(df)} rows.")
        return df

    except Exception as e:
        log_error(f"Error calculating technical indicators: {e}", exc_info=True)
        return None

# --- Position Sizing Function (Decorated) ---
@api_retry_decorator
def calculate_position_size(exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, stop_loss_price: float, risk_perc: float) -> Optional[float]:
    """Calculates position size based on risk percentage and stop-loss distance. Retries on network errors."""
    try:
        log_debug(f"Calculating position size: Symbol={trading_symbol}, Price={current_price}, SL={stop_loss_price}, Risk={risk_perc*100}%")
        # Fetch account balance (This call is retried by the decorator)
        balance = exchange_instance.fetch_balance()
        market = exchange_instance.market(trading_symbol)

        # Determine quote currency
        quote_currency = market.get('quote')
        if not quote_currency:
             # Fallback for symbols like BTC/USDT:USDT
             parts = trading_symbol.split('/')
             if len(parts) > 1:
                 quote_currency = parts[1].split(':')[0]
        if not quote_currency:
             log_error(f"Could not determine quote currency for {trading_symbol}.")
             return None
        log_debug(f"Quote currency: {quote_currency}")

        # Find available balance in quote currency (handle variations in balance structure)
        available_balance = 0.0
        # Try common structures first
        if quote_currency in balance.get('free', {}):
            available_balance = float(balance['free'][quote_currency])
        elif 'free' in balance and isinstance(balance['free'], dict) and quote_currency in balance['free']:
              available_balance = float(balance['free'][quote_currency])
        # Check specific currency entry if 'free' wasn't top-level dict
        elif quote_currency in balance:
             available_balance = float(balance[quote_currency].get('free', 0.0))

        # Add more specific checks based on exchange if needed (e.g., futures wallets 'USDT' vs 'info')
        # Example for Bybit USDT perpetual: balance['USDT']['availableBalance'] might be relevant
        # if exchange_instance.id == 'bybit' and market.get('linear'):
        #     available_balance = float(balance.get('USDT', {}).get('availableBalance', available_balance))


        if available_balance <= 0:
            log_error(f"No available {quote_currency} balance ({available_balance}) found for trading. Check balance structure/funds. Balance details: {balance.get(quote_currency, 'N/A')}")
            return None
        log_info(f"Available balance ({quote_currency}): {available_balance:.{price_precision_digits}f}")

        # Calculate risk amount
        risk_amount = available_balance * risk_perc
        log_info(f"Risk per trade ({risk_perc*100:.2f}%): {risk_amount:.{price_precision_digits}f} {quote_currency}")

        # Calculate price difference for SL
        price_diff = abs(current_price - stop_loss_price)

        if price_diff <= min_tick / 2: # Use half tick as threshold
            log_error(f"Stop-loss price {stop_loss_price:.{price_precision_digits}f} is too close to current price {current_price:.{price_precision_digits}f} (Diff: {price_diff}). Cannot calculate size.")
            return None
        log_debug(f"Price difference for SL: {price_diff:.{price_precision_digits}f}")

        # Calculate quantity
        quantity = risk_amount / price_diff
        log_debug(f"Calculated raw quantity: {quantity:.{amount_precision_digits+4}f}") # Show more precision initially

        # Adjust for precision and limits
        try:
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, quantity))
            log_debug(f"Quantity adjusted for precision: {quantity_adjusted:.{amount_precision_digits}f}")
        except ccxt.ExchangeError as precision_error:
             log_warning(f"Could not use exchange.amount_to_precision: {precision_error}. Using raw quantity rounded.")
             # Fallback: round manually (less accurate than exchange method)
             quantity_adjusted = round(quantity, amount_precision_digits)


        if quantity_adjusted <= 0:
            log_error(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} is zero or negative.")
            return None

        # Check limits (min/max amount, min cost)
        limits = market.get('limits', {})
        min_amount = limits.get('amount', {}).get('min')
        max_amount = limits.get('amount', {}).get('max')
        min_cost = limits.get('cost', {}).get('min')

        log_debug(f"Market Limits: Min Qty={min_amount}, Max Qty={max_amount}, Min Cost={min_cost}")

        if min_amount is not None and quantity_adjusted < min_amount:
            log_error(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} is below min amount {min_amount}.")
            return None
        if max_amount is not None and quantity_adjusted > max_amount:
            log_warning(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} exceeds max amount {max_amount}. Capping.")
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, max_amount))

        estimated_cost = quantity_adjusted * current_price
        if min_cost is not None and estimated_cost < min_cost:
             log_error(f"Estimated order cost ({estimated_cost:.{price_precision_digits}f}) is below min cost {min_cost}.")
             return None

        # Sanity check cost vs balance
        cost_buffer = 0.995 # Use 99.5% of balance as limit for cost check
        if estimated_cost > available_balance * cost_buffer:
             log_error(f"Estimated cost ({estimated_cost:.{price_precision_digits}f}) exceeds {cost_buffer*100:.1f}% of available balance ({available_balance:.{price_precision_digits}f}). Reduce risk % or add funds.")
             return None

        log_info(f"{NEON_GREEN}Position size calculated: {quantity_adjusted:.{amount_precision_digits}f} {market.get('base', '')}{RESET} (Risking ~{risk_amount:.2f} {quote_currency})")
        return quantity_adjusted

    # Handle non-retryable errors specifically
    except ccxt.AuthenticationError as e:
         log_error(f"Authentication error during position size calculation (fetching balance): {e}")
         return None
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange error during position size calculation: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error calculating position size: {e}", exc_info=True)
        # If it was a retryable error, the decorator would have logged and raised it.
        return None


# --- Order Placement Functions (Decorated) ---

# Helper for cancellation with retry
@api_retry_decorator
def cancel_order_with_retry(exchange_instance: ccxt.Exchange, order_id: str, trading_symbol: str):
    """Cancels an order by ID with retry logic."""
    if not order_id:
        log_debug("No order ID provided to cancel.")
        return # Nothing to cancel

    log_info(f"Attempting to cancel order ID: {order_id} for {trading_symbol}...")
    if not SIMULATION_MODE:
        exchange_instance.cancel_order(order_id, trading_symbol)
        log_info(f"Order {order_id} cancellation request sent.")
    else:
        log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")


@api_retry_decorator
def place_market_order(exchange_instance: ccxt.Exchange, trading_symbol: str, side: str, amount: float, reduce_only: bool = False) -> Optional[Dict[str, Any]]:
    """Places a market order ('buy' or 'sell') with retry logic. Optionally reduceOnly."""
    if side not in ['buy', 'sell']:
        log_error(f"Invalid order side: '{side}'.")
        return None
    if amount <= 0:
        log_error(f"Invalid order amount: {amount}.")
        return None

    try:
        market = exchange_instance.market(trading_symbol)
        base_currency: str = market.get('base', '')
        quote_currency: str = market.get('quote', '')
        amount_formatted = float(exchange_instance.amount_to_precision(trading_symbol, amount))

        log_info(f"Attempting to place {side.upper()} market order for {amount_formatted:.{amount_precision_digits}f} {base_currency} {'(ReduceOnly)' if reduce_only else ''}...")

        params = {}
        # Check if exchange explicitly supports reduceOnly for market orders, common for futures/swap
        if reduce_only and exchange_instance.has.get('reduceOnly'):
             # Check if market type likely supports it (swap/future)
             if market.get('swap') or market.get('future') or market.get('contract'):
                  params['reduceOnly'] = True
                  log_debug("Applying 'reduceOnly=True' to market order.")
             else:
                  log_warning("ReduceOnly requested but market type is likely SPOT. Ignoring reduceOnly param for market order.")


        if SIMULATION_MODE:
            log_warning("!!! SIMULATION: Market order placement skipped.")
            # Create a realistic dummy order response
            sim_price = 0.0
            try: # Fetch ticker safely
                 # Fetching ticker can also fail, apply retry decorator? Maybe overkill here.
                 ticker = exchange_instance.fetch_ticker(trading_symbol)
                 sim_price = ticker['last'] if ticker and 'last' in ticker else 0.0
            except Exception as ticker_e:
                log_warning(f"Could not fetch ticker for simulation price: {ticker_e}")

            order_id = f'sim_market_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
            sim_cost = amount_formatted * sim_price if sim_price > 0 else 0.0
            order = {
                'id': order_id, 'clientOrderId': order_id, 'timestamp': int(time.time() * 1000),
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'closed', # Assume instant fill
                'symbol': trading_symbol, 'type': 'market', 'timeInForce': 'IOC',
                'side': side, 'price': sim_price, 'average': sim_price,
                'amount': amount_formatted, 'filled': amount_formatted, 'remaining': 0.0,
                'cost': sim_cost, 'fee': None, 'info': {'simulated': True, 'reduceOnly': params.get('reduceOnly', False)}
            }
        else:
            # --- LIVE TRADING ---
            log_warning(f"!!! LIVE MODE: Placing real market order {'(ReduceOnly)' if params.get('reduceOnly') else ''}.")
            order = exchange_instance.create_market_order(trading_symbol, side, amount_formatted, params=params)
            # --- END LIVE TRADING ---

        log_info(f"{side.capitalize()} market order request processed {'(Simulated)' if SIMULATION_MODE else ''}.")
        order_id: Optional[str] = order.get('id')
        order_status: Optional[str] = order.get('status')
        # Use 'average' if available (filled price), fallback to 'price' (less reliable for market)
        order_price: Optional[float] = order.get('average', order.get('price'))
        order_filled: Optional[float] = order.get('filled')
        order_cost: Optional[float] = order.get('cost')

        price_str: str = f"{order_price:.{price_precision_digits}f}" if isinstance(order_price, (int, float)) else "N/A"
        filled_str: str = f"{order_filled:.{amount_precision_digits}f}" if isinstance(order_filled, (int, float)) else "N/A"
        cost_str: str = f"{order_cost:.{price_precision_digits}f}" if isinstance(order_cost, (int, float)) else "N/A"

        log_info(f"Order Result | ID: {order_id or 'N/A'}, Status: {order_status or 'N/A'}, Avg Price: {price_str}, "
                    f"Filled: {filled_str} {base_currency}, Cost: {cost_str} {quote_currency}")
        # Add short delay after placing order to allow exchange processing / state update
        time.sleep(1.5) # Increased delay slightly
        return order

    # Handle specific non-retryable errors
    except ccxt.InsufficientFunds as e:
        log_error(f"Insufficient funds for {side} {amount} {trading_symbol}. Error: {e}")
        return None
    except ccxt.OrderNotFound as e: # Can happen if order rejected immediately
        log_error(f"OrderNotFound error placing {side} {amount} {trading_symbol}. Likely rejected immediately. Error: {e}")
        return None
    except ccxt.InvalidOrder as e:
         log_error(f"Invalid market order parameters for {side} {amount} {trading_symbol}: {e}")
         return None
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error placing {side} market order for {trading_symbol}: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error placing {side} market order: {e}", exc_info=True)
        return None


# Note: Applying retry to the *entire* SL/TP function can be complex if one order
# succeeds and the other fails, leading to duplicate orders on retry.
# A more granular approach (retrying individual create_order calls within) might be better,
# but for simplicity, we apply it to the whole function here. Be cautious in live mode.
@api_retry_decorator
def place_sl_tp_orders(exchange_instance: ccxt.Exchange, trading_symbol: str, position_side: str, quantity: float, sl_price: float, tp_price: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Places stop-loss (stop-market preferred) and take-profit (limit) orders.
    Uses reduceOnly. Retries the whole placement process on network errors.
    """
    sl_order: Optional[Dict[str, Any]] = None
    tp_order: Optional[Dict[str, Any]] = None

    if quantity <= 0 or position_side not in ['long', 'short']:
        log_error(f"Invalid input for SL/TP placement: Qty={quantity}, Side='{position_side}'")
        return None, None

    try:
        market = exchange_instance.market(trading_symbol)
        close_side = 'sell' if position_side == 'long' else 'buy'
        qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, quantity))
        sl_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, sl_price))
        tp_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, tp_price))

        # Check exchange capabilities
        # ReduceOnly is often a param, not a separate capability check in ccxt `has`
        has_reduce_only_param = True # Assume most modern exchanges support it in params
        has_stop_market = exchange_instance.has.get('createStopMarketOrder', False)
        has_stop_limit = exchange_instance.has.get('createStopLimitOrder', False)
        has_limit = exchange_instance.has.get('createLimitOrder', True) # Assume basic limit exists

        sl_params = {'reduceOnly': True} if has_reduce_only_param else {}
        tp_params = {'reduceOnly': True} if has_reduce_only_param else {}

        # --- Place Stop-Loss Order ---
        log_info(f"Attempting SL order: {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} @ trigger ~{sl_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in sl_params else ''}")
        sl_order_type = None
        sl_order_price = None # Only needed for limit types

        # Prefer stopMarket if available via createOrder
        # Note: Some exchanges require create_stop_market_order explicitly
        if has_stop_market or 'stopMarket' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):
            sl_order_type = 'stopMarket'
            # CCXT standard uses 'stopPrice' in params for stop orders via create_order
            sl_params['stopPrice'] = sl_price_formatted
            log_debug("Using stopMarket type for SL via createOrder.")
        elif has_stop_limit or 'stopLimit' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):
            sl_order_type = 'stopLimit'
            # StopLimit requires a trigger (stopPrice) and a limit price.
            # Set limit price slightly worse than trigger to increase fill chance
            limit_offset = abs(tp_price_formatted - sl_price_formatted) * 0.1 # 10% of TP-SL range as offset
            limit_offset = max(limit_offset, min_tick * 5) # Ensure offset is at least a few ticks
            sl_limit_price = sl_price_formatted - limit_offset if close_side == 'sell' else sl_price_formatted + limit_offset
            sl_order_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price))

            sl_params['stopPrice'] = sl_price_formatted
            sl_params['price'] = sl_order_price # The limit price for the order placed after trigger
            log_warning(f"Using stopLimit type for SL. Trigger: {sl_price_formatted:.{price_precision_digits}f}, Limit: {sl_order_price:.{price_precision_digits}f}.")
        else:
            log_error(f"Exchange {exchange_instance.id} supports neither stopMarket nor stopLimit orders via CCXT createOrder. Cannot place automated SL.")
            # Continue to try placing TP

        if sl_order_type:
            if SIMULATION_MODE:
                log_warning("!!! SIMULATION: Stop-loss order placement skipped.")
                sl_order_id = f'sim_sl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                sl_order = {
                    'id': sl_order_id, 'status': 'open', 'symbol': trading_symbol,
                    'type': sl_order_type, 'side': close_side, 'amount': qty_formatted,
                    'price': sl_order_price, # Limit price if stopLimit
                    'stopPrice': sl_params.get('stopPrice'), # Trigger price
                    'average': None, 'filled': 0.0, 'remaining': qty_formatted, 'cost': 0.0,
                    'info': {'simulated': True, 'reduceOnly': sl_params.get('reduceOnly', False)}
                }
            else:
                # --- LIVE TRADING ---
                log_warning(f"!!! LIVE MODE: Placing real {sl_order_type} stop-loss order.")
                # Use create_order as it handles various stop types with params
                sl_order = exchange_instance.create_order(
                    symbol=trading_symbol,
                    type=sl_order_type,
                    side=close_side,
                    amount=qty_formatted,
                    price=sl_order_price, # Required only for limit types (like stopLimit)
                    params=sl_params
                )
                log_info(f"Stop-loss order request processed. ID: {sl_order.get('id', 'N/A')}")
                time.sleep(0.75) # Small delay after live order
                # --- END LIVE TRADING ---

        # --- Place Take-Profit Order ---
        log_info(f"Attempting TP order: {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} @ limit {tp_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in tp_params else ''}")
        tp_order_type = 'limit' # Standard limit order for TP

        if has_limit: # Check if limit orders are supported (should almost always be true)
             if SIMULATION_MODE:
                 log_warning("!!! SIMULATION: Take-profit order placement skipped.")
                 tp_order_id = f'sim_tp_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                 tp_order = {
                     'id': tp_order_id, 'status': 'open', 'symbol': trading_symbol,
                     'type': tp_order_type, 'side': close_side, 'amount': qty_formatted,
                     'price': tp_price_formatted, 'average': None,
                     'filled': 0.0, 'remaining': qty_formatted, 'cost': 0.0,
                     'info': {'simulated': True, 'reduceOnly': tp_params.get('reduceOnly', False)}
                 }
             else:
                # --- LIVE TRADING ---
                 log_warning(f"!!! LIVE MODE: Placing real {tp_order_type} take-profit order.")
                 tp_order = exchange_instance.create_limit_order(
                     symbol=trading_symbol,
                     side=close_side,
                     amount=qty_formatted,
                     price=tp_price_formatted,
                     params=tp_params
                 )
                 log_info(f"Take-profit order request processed. ID: {tp_order.get('id', 'N/A')}")
                 time.sleep(0.75) # Small delay after live order
                 # --- END LIVE TRADING ---
        else:
             log_error(f"Exchange {exchange_instance.id} does not support limit orders. Cannot place take-profit.")
             tp_order = None


        # Final check and warnings if one failed
        sl_ok = sl_order and sl_order.get('id')
        tp_ok = tp_order and tp_order.get('id')
        if sl_ok and not tp_ok: log_warning("SL placed, but TP failed.")
        elif not sl_ok and tp_ok: log_warning("TP placed, but SL failed. Position unprotected!")
        elif not sl_ok and not tp_ok: log_error("Both SL and TP order placements failed.")

        return sl_order, tp_order

    # Handle specific non-retryable errors
    except ccxt.InvalidOrder as e:
        log_error(f"Invalid order parameters placing SL/TP: {e}")
        # Return whatever might have succeeded before the error
        return sl_order, tp_order
    except ccxt.InsufficientFunds as e: # Should not happen with reduceOnly, but possible
         log_error(f"Insufficient funds error during SL/TP placement (unexpected): {e}")
         return sl_order, tp_order
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error placing SL/TP orders: {e}")
        return sl_order, tp_order
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error in place_sl_tp_orders: {e}", exc_info=True)
        # Return None, None to indicate total failure if an exception occurred
        return None, None


# --- Position and Order Check Function (Decorated) ---
@api_retry_decorator
def check_position_and_orders(exchange_instance: ccxt.Exchange, trading_symbol: str) -> bool:
    """
    Checks if tracked SL/TP orders are still open. If not, assumes filled,
    cancels the other order, resets local state, and returns True. Retries on network errors.
    Uses fetch_open_orders.
    """
    global position
    if position['status'] is None:
        return False # No active position to check

    log_debug(f"Checking open orders vs local state for {trading_symbol}...")
    position_reset_flag = False
    try:
        # Fetch Open Orders for the specific symbol (retried by decorator)
        open_orders = exchange_instance.fetch_open_orders(trading_symbol)
        log_debug(f"Found {len(open_orders)} open orders for {trading_symbol}.")

        sl_order_id = position.get('sl_order_id')
        tp_order_id = position.get('tp_order_id')

        if not sl_order_id and not tp_order_id:
             # This state can occur if SL/TP placement failed initially or after a restart
             # without proper state restoration of order IDs.
             log_warning(f"In {position['status']} position but no SL/TP order IDs tracked. Cannot verify closure via orders.")
             # Alternative: Try fetching current position size from exchange here?
             # `fetch_positions` is often needed for futures/swaps, but can be complex/inconsistent.
             # Sticking to order-based check for now.
             return False # Cannot determine closure reliably

        # Check if tracked orders are still present in the fetched open orders
        open_order_ids = {order.get('id') for order in open_orders if order.get('id')}
        sl_found_open = sl_order_id in open_order_ids
        tp_found_open = tp_order_id in open_order_ids

        log_debug(f"Tracked SL ({sl_order_id or 'None'}) open: {sl_found_open}. Tracked TP ({tp_order_id or 'None'}) open: {tp_found_open}.")

        # --- Logic for State Reset ---
        order_to_cancel_id = None
        assumed_close_reason = None

        # If SL was tracked but is no longer open:
        if sl_order_id and not sl_found_open:
            log_info(f"{NEON_YELLOW}Stop-loss order {sl_order_id} no longer open. Assuming position closed via SL.{RESET}")
            position_reset_flag = True
            order_to_cancel_id = tp_order_id # Attempt to cancel the TP order
            assumed_close_reason = "SL"

        # Else if TP was tracked but is no longer open:
        elif tp_order_id and not tp_found_open:
            log_info(f"{NEON_GREEN}Take-profit order {tp_order_id} no longer open. Assuming position closed via TP.{RESET}")
            position_reset_flag = True
            order_to_cancel_id = sl_order_id # Attempt to cancel the SL order
            assumed_close_reason = "TP"

        # If a reset was triggered:
        if position_reset_flag:
            # Attempt to cancel the other order using the retry helper
            if order_to_cancel_id:
                 log_info(f"Attempting to cancel leftover {'TP' if assumed_close_reason == 'SL' else 'SL'} order {order_to_cancel_id}...")
                 try:
                     # Use the decorated cancel helper
                     cancel_order_with_retry(exchange_instance, order_to_cancel_id, trading_symbol)
                 except ccxt.OrderNotFound:
                     log_info(f"Order {order_to_cancel_id} was already closed/cancelled.")
                 except Exception as e:
                     # Error logged by cancel_order_with_retry or its decorator
                     log_error(f"Failed to cancel leftover order {order_to_cancel_id} after retries. Manual check advised.", exc_info=False) # Don't need full trace here
            else:
                 log_debug(f"No corresponding {'TP' if assumed_close_reason == 'SL' else 'SL'} order ID was tracked or provided to cancel.")


            log_info("Resetting local position state.")
            # Reset all relevant fields
            position.update({
                'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                'stop_loss': None, 'take_profit': None, 'entry_time': None,
                'sl_order_id': None, 'tp_order_id': None,
                # Reset trailing stop fields as well
                'highest_price_since_entry': None, 'lowest_price_since_entry': None,
                'current_trailing_sl_price': None
            })
            save_position_state() # Save the reset state
            display_position_status(position, price_precision_digits, amount_precision_digits) # Show updated status
            return True # Indicate position was reset

        # If both orders are still open, or only one was tracked and it's still open
        log_debug("Position state appears consistent with tracked open orders.")
        return False # Position was not reset

    # Handle specific non-retryable errors from fetch_open_orders if they bypass decorator
    except ccxt.AuthenticationError as e:
         log_error(f"Authentication error checking position/orders: {e}")
         return False
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error checking position/orders: {e}")
        return False
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error checking position/orders: {e}", exc_info=True)
        # Let retry handler manage retryable exceptions
        return False # Assume state is uncertain if error occurred

# --- Trailing Stop Loss Update Function (From Snippet 2, Adapted) ---
def update_trailing_stop(exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, last_atr: float):
    """Checks and updates the trailing stop loss if conditions are met."""
    global position
    if (position['status'] is None or
        not enable_trailing_stop or
        position['sl_order_id'] is None or # Need an active SL order to trail
        last_atr <= 0): # Cannot trail if no position, TSL disabled, no SL order, or invalid ATR
        if enable_trailing_stop and position['status'] is not None and last_atr <= 0:
            log_warning("Cannot perform trailing stop check: Invalid ATR value.")
        elif enable_trailing_stop and position['status'] is not None and position['sl_order_id'] is None:
            log_warning("Cannot perform trailing stop check: No active SL order ID tracked.")
        return

    log_debug(f"Checking trailing stop. Current SL Order: {position['sl_order_id']}, Current Tracked TSL Price: {position.get('current_trailing_sl_price')}")

    new_potential_tsl = None
    activation_threshold_met = False
    # Use current_trailing_sl_price if set, otherwise use the initial stop_loss from state
    current_effective_sl = position.get('current_trailing_sl_price') or position.get('stop_loss')

    if current_effective_sl is None:
        log_warning("Cannot check TSL improvement: current effective SL price is None.")
        return

    if position['status'] == 'long':
        entry_price = position.get('entry_price', current_price) # Fallback to current if entry somehow missing
        # Update highest price seen since entry
        highest_seen = position.get('highest_price_since_entry', entry_price)
        if current_price > highest_seen:
            position['highest_price_since_entry'] = current_price
            highest_seen = current_price
            # Save state less frequently, e.g., only when TSL updates, to reduce disk I/O
            # save_position_state() # Persist the new high - potentially too frequent

        # Check activation threshold (price must move ATR * activation_multiplier above entry)
        activation_price = entry_price + (last_atr * trailing_stop_activation_atr_multiplier)

        if highest_seen > activation_price:
            activation_threshold_met = True
            # Calculate potential new trailing stop based on the highest price seen
            new_potential_tsl = highest_seen - (last_atr * trailing_stop_atr_multiplier)
            log_debug(f"Long TSL Activation Met. Highest: {highest_seen:.{price_precision_digits}f} > Activation: {activation_price:.{price_precision_digits}f}. Potential New TSL: {new_potential_tsl:.{price_precision_digits}f}")
        else:
             log_debug(f"Long TSL Activation NOT Met. Highest: {highest_seen:.{price_precision_digits}f} <= Activation: {activation_price:.{price_precision_digits}f}")


    elif position['status'] == 'short':
        entry_price = position.get('entry_price', current_price)
        # Update lowest price seen since entry
        lowest_seen = position.get('lowest_price_since_entry', entry_price)
        if current_price < lowest_seen:
            position['lowest_price_since_entry'] = current_price
            lowest_seen = current_price
            # save_position_state() # Persist the new low - potentially too frequent

        # Check activation threshold
        activation_price = entry_price - (last_atr * trailing_stop_activation_atr_multiplier)

        if lowest_seen < activation_price:
            activation_threshold_met = True
            # Calculate potential new trailing stop
            new_potential_tsl = lowest_seen + (last_atr * trailing_stop_atr_multiplier)
            log_debug(f"Short TSL Activation Met. Lowest: {lowest_seen:.{price_precision_digits}f} < Activation: {activation_price:.{price_precision_digits}f}. Potential New TSL: {new_potential_tsl:.{price_precision_digits}f}")
        else:
            log_debug(f"Short TSL Activation NOT Met. Lowest: {lowest_seen:.{price_precision_digits}f} >= Activation: {activation_price:.{price_precision_digits}f}")


    if activation_threshold_met and new_potential_tsl is not None:
        # --- Check if the new TSL is an improvement ---
        should_update_tsl = False
        # Format potential new TSL to market precision for comparison
        new_tsl_formatted = float(exchange_instance.price_to_precision(trading_symbol, new_potential_tsl))

        if position['status'] == 'long':
             # New TSL must be strictly higher than current effective SL
             # And must be below the current price (with a small buffer maybe?) to avoid immediate stop-out
             if new_tsl_formatted > current_effective_sl and new_tsl_formatted < current_price:
                 should_update_tsl = True
                 log_debug(f"Long TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) > Current ({current_effective_sl:.{price_precision_digits}f}) AND < Price ({current_price:.{price_precision_digits}f}) -> OK")
             else:
                 log_debug(f"Long TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) vs Current ({current_effective_sl:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}) -> NO UPDATE")

        elif position['status'] == 'short':
             # New TSL must be strictly lower than current effective SL
             # And must be above the current price
             if new_tsl_formatted < current_effective_sl and new_tsl_formatted > current_price:
                 should_update_tsl = True
                 log_debug(f"Short TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) < Current ({current_effective_sl:.{price_precision_digits}f}) AND > Price ({current_price:.{price_precision_digits}f}) -> OK")
             else:
                  log_debug(f"Short TSL check: New ({new_tsl_formatted:.{price_precision_digits}f}) vs Current ({current_effective_sl:.{price_precision_digits}f}) vs Price ({current_price:.{price_precision_digits}f}) -> NO UPDATE")

        if should_update_tsl:
            log_info(f"{NEON_YELLOW}Trailing Stop Update Triggered! New target SL: {new_tsl_formatted:.{price_precision_digits}f}{RESET}")

            # --- Modify Existing SL Order (Cancel old, Place new) ---
            old_sl_id = position['sl_order_id']
            if not old_sl_id:
                log_error("TSL triggered but no old SL order ID found in state. Cannot modify.")
                return

            try:
                # 1. Cancel Old SL Order (using retry helper)
                cancel_order_with_retry(exchange_instance, old_sl_id, trading_symbol)
                log_info(f"Old SL order {old_sl_id} cancellation request sent/simulated.")
                # Short delay to allow cancellation processing
                time.sleep(1.0)

                # 2. Place New SL Order (use place_sl_tp_orders, but only use the SL part)
                # We need the original TP price to calculate stopLimit offset if needed, get from state
                original_tp_price = position.get('take_profit', current_price * (1 + 0.05) if position['status'] == 'long' else current_price * (1 - 0.05)) # Fallback TP
                log_info(f"Placing new TSL order at {new_tsl_formatted:.{price_precision_digits}f}")

                # Call place_sl_tp_orders but only care about the sl_order result
                # Pass a dummy TP price far away, or the original TP if available
                new_sl_order, _ = place_sl_tp_orders(
                    exchange_instance=exchange_instance,
                    trading_symbol=trading_symbol,
                    position_side=position['status'],
                    quantity=position['quantity'],
                    sl_price=new_tsl_formatted,
                    tp_price=original_tp_price # Pass original TP, place_sl_tp should only place SL if needed
                    # NOTE: place_sl_tp_orders currently tries to place both. Need modification
                    # or a dedicated place_sl_order function.
                    # Let's modify place_sl_tp_orders to accept optional placement flags.
                    # ---> Modification needed in place_sl_tp_orders or use direct create_order here.

                    # --- Alternative: Direct Placement (Simplified) ---
                    # Replicating the SL placement logic here:
                )
                new_sl_order_direct = None # Placeholder for direct placement result
                try:
                    close_side = 'sell' if position['status'] == 'long' else 'buy'
                    qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, position['quantity']))
                    has_reduce_only_param = True
                    has_stop_market = exchange_instance.has.get('createStopMarketOrder', False)
                    has_stop_limit = exchange_instance.has.get('createStopLimitOrder', False)
                    new_sl_params = {'reduceOnly': True} if has_reduce_only_param else {}
                    new_sl_order_type = None
                    new_sl_order_price = None

                    if has_stop_market or 'stopMarket' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):
                        new_sl_order_type = 'stopMarket'
                        new_sl_params['stopPrice'] = new_tsl_formatted
                    elif has_stop_limit or 'stopLimit' in exchange_instance.options.get('createOrder', {}).get('stopTypes', []):
                        new_sl_order_type = 'stopLimit'
                        limit_offset = abs(original_tp_price - new_tsl_formatted) * 0.1
                        limit_offset = max(limit_offset, min_tick * 5)
                        sl_limit_price = new_tsl_formatted - limit_offset if close_side == 'sell' else new_tsl_formatted + limit_offset
                        new_sl_order_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price))
                        new_sl_params['stopPrice'] = new_tsl_formatted
                        new_sl_params['price'] = new_sl_order_price
                    else:
                         raise ccxt.NotSupported("Exchange supports neither stopMarket nor stopLimit for TSL.")

                    log_info(f"Attempting direct placement of new TSL order ({new_sl_order_type})")
                    if SIMULATION_MODE:
                        log_warning("!!! SIMULATION: New TSL order placement skipped.")
                        new_sl_order_id = f'sim_tsl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                        new_sl_order_direct = {'id': new_sl_order_id, 'status': 'open', 'info': {'simulated': True}}
                    else:
                        log_warning(f"!!! LIVE MODE: Placing real {new_sl_order_type} TSL order.")
                        # Apply retry to this specific call
                        @api_retry_decorator
                        def place_new_tsl_order_direct():
                            return exchange_instance.create_order(
                                symbol=trading_symbol, type=new_sl_order_type, side=close_side,
                                amount=qty_formatted, price=new_sl_order_price, params=new_sl_params
                            )
                        new_sl_order_direct = place_new_tsl_order_direct()

                    # --- End Direct Placement ---


                    if new_sl_order_direct and new_sl_order_direct.get('id'):
                        new_id = new_sl_order_direct['id']
                        log_info(f"New trailing SL order placed successfully: ID {new_id}")
                        # Update position state with new SL info
                        position['stop_loss'] = new_tsl_formatted # Update the main SL reference
                        position['sl_order_id'] = new_id
                        position['current_trailing_sl_price'] = new_tsl_formatted # Mark TSL active price
                        save_position_state() # Save the successful update
                    else:
                        log_error(f"Failed to place new trailing SL order after cancelling old one {old_sl_id}. POSITION MAY BE UNPROTECTED.")
                        position['sl_order_id'] = None # Mark SL as lost
                        position['current_trailing_sl_price'] = None
                        save_position_state()

                except Exception as place_e:
                     # Error logged by retry decorator or generic handler
                     log_error(f"Error placing new trailing SL order: {place_e}. POSITION MAY BE UNPROTECTED.", exc_info=True)
                     position['sl_order_id'] = None
                     position['current_trailing_sl_price'] = None
                     save_position_state()


            except ccxt.OrderNotFound:
                log_warning(f"Old SL order {old_sl_id} not found during TSL update (might have been filled/cancelled already).")
                # If SL filled, check_position_and_orders should handle the reset.
                # Mark TSL as inactive for this cycle if SL ID is now gone.
                position['current_trailing_sl_price'] = None
                position['sl_order_id'] = None # Ensure SL ID is cleared if not found
                save_position_state() # Save the cleared state
            except Exception as cancel_e:
                # Error logged by cancel_order_with_retry or its decorator
                log_error(f"Error cancelling old SL order {old_sl_id} during TSL update: {cancel_e}. Aborting TSL placement.", exc_info=False)
                # Do not proceed to place new SL if cancellation failed unexpectedly.

# --- Main Trading Loop ---
log_info(f"Initializing trading bot for {symbol} on {timeframe}...")
# Load position state ONCE at startup
load_position_state()
log_info(f"Risk per trade: {risk_percentage*100:.2f}%")
log_info(f"Bot check interval: {sleep_interval_seconds} seconds ({sleep_interval_seconds/60:.1f} minutes)")
log_info(f"{NEON_YELLOW}Press Ctrl+C to stop the bot gracefully.{RESET}")

while True:
    try:
        cycle_start_time: pd.Timestamp = pd.Timestamp.now(tz='UTC')
        print_cycle_divider(cycle_start_time)

        # 1. Check Position/Order Consistency FIRST
        position_was_reset = check_position_and_orders(exchange, symbol)
        display_position_status(position, price_precision_digits, amount_precision_digits) # Display status after check
        if position_was_reset:
             log_info("Position state reset by order check. Proceeding to check for new entries.")
             # Continue loop to immediately check for signals

        # 2. Fetch Fresh OHLCV Data
        ohlcv_df: Optional[pd.DataFrame] = fetch_ohlcv_data(exchange, symbol, timeframe, limit_count=data_limit)
        if ohlcv_df is None or ohlcv_df.empty:
            log_warning(f"Could not fetch valid OHLCV data. Waiting...")
            neon_sleep_timer(sleep_interval_seconds)
            continue

        # 3. Calculate Technical Indicators (Conditionally include ATR, Vol MA)
        needs_atr = enable_atr_sl_tp or enable_trailing_stop
        needs_vol_ma = entry_volume_confirmation_enabled
        stoch_params = {'k': stoch_k, 'd': stoch_d, 'smooth_k': stoch_smooth_k}
        df_with_indicators: Optional[pd.DataFrame] = calculate_technical_indicators(
            ohlcv_df.copy(), # Use copy
            rsi_len=rsi_length, stoch_params=stoch_params,
            calc_atr=needs_atr, atr_len=atr_length,
            calc_vol_ma=needs_vol_ma, vol_ma_len=entry_volume_ma_length
        )
        if df_with_indicators is None or df_with_indicators.empty:
             log_warning(f"Indicator calculation failed. Waiting...")
             neon_sleep_timer(sleep_interval_seconds)
             continue

        # 4. Get Latest Data and Indicator Values
        if len(df_with_indicators) < 2:
            log_warning("Not enough data points after indicator calculation. Waiting...")
            neon_sleep_timer(sleep_interval_seconds)
            continue

        latest_data: pd.Series = df_with_indicators.iloc[-1]
        # Construct indicator column names
        rsi_col_name: str = f'RSI_{rsi_length}'
        stoch_k_col_name: str = f'STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}'
        stoch_d_col_name: str = f'STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}'
        # ATR column from pandas_ta is typically ATRr_LENGTH (raw ATR)
        atr_col_name: str = f'ATRr_{atr_length}' if needs_atr else None
        vol_ma_col_name: str = f'VOL_MA_{entry_volume_ma_length}' if needs_vol_ma else None

        # Check required columns exist
        required_base_cols: List[str] = ['close', 'high', 'low', 'open', 'volume']
        required_indicator_cols: List[str] = [rsi_col_name, stoch_k_col_name, stoch_d_col_name]
        if needs_atr: required_indicator_cols.append(atr_col_name)
        if needs_vol_ma and 'volume' in df_with_indicators.columns: required_indicator_cols.append(vol_ma_col_name)

        all_required_cols = required_base_cols + [col for col in required_indicator_cols if col is not None]
        missing_cols = [col for col in all_required_cols if col not in df_with_indicators.columns]

        if missing_cols:
            log_error(f"Required columns missing in DataFrame: {missing_cols}. Check config/data. Available: {df_with_indicators.columns.tolist()}")
            neon_sleep_timer(sleep_interval_seconds)
            continue

        # Extract latest values safely, checking for NaNs
        try:
            current_price: float = float(latest_data['close'])
            current_high: float = float(latest_data['high'])
            current_low: float = float(latest_data['low'])
            last_rsi: float = float(latest_data[rsi_col_name])
            last_stoch_k: float = float(latest_data[stoch_k_col_name])
            last_stoch_d: float = float(latest_data[stoch_d_col_name])

            last_atr: Optional[float] = None
            if needs_atr:
                atr_val = latest_data.get(atr_col_name)
                if pd.notna(atr_val):
                    last_atr = float(atr_val)
                else:
                     log_warning(f"ATR value is NaN for the latest candle. ATR-based logic will be skipped.")


            current_volume: Optional[float] = None
            last_volume_ma: Optional[float] = None
            if needs_vol_ma and 'volume' in latest_data:
                 vol_val = latest_data.get('volume')
                 vol_ma_val = latest_data.get(vol_ma_col_name)
                 if pd.notna(vol_val): current_volume = float(vol_val)
                 if pd.notna(vol_ma_val): last_volume_ma = float(vol_ma_val)
                 if current_volume is None or last_volume_ma is None:
                     log_warning("Volume or Volume MA is NaN. Volume confirmation may be skipped.")

            # Final check for essential NaNs
            essential_values = [current_price, last_rsi, last_stoch_k, last_stoch_d]
            if any(pd.isna(v) for v in essential_values):
                 raise ValueError("Essential indicator value is NaN.")

        except (KeyError, ValueError, TypeError) as e:
             log_error(f"Error extracting latest data or NaN found: {e}. Data: {latest_data.to_dict()}", exc_info=True)
             neon_sleep_timer(sleep_interval_seconds)
             continue

        # Display Market Stats
        display_market_stats(current_price, last_rsi, last_stoch_k, last_stoch_d, last_atr, price_precision_digits)

        # 5. Identify Order Blocks
        bullish_ob, bearish_ob = identify_potential_order_block(
            df_with_indicators,
            vol_thresh_mult=ob_volume_threshold_multiplier,
            lookback_len=ob_lookback
        )
        display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)


        # 6. Apply Trading Logic

        # --- Check if ALREADY IN A POSITION ---
        if position['status'] is not None:
            log_info(f"Currently in {position['status'].upper()} position.")

            # A. Check Trailing Stop Logic (only if enabled and ATR available)
            if enable_trailing_stop and last_atr is not None and last_atr > 0:
                log_debug("Checking trailing stop condition...")
                update_trailing_stop(exchange, symbol, current_price, last_atr)
                # Re-check position status in case TSL caused an exit (though check_pos should catch it next cycle)
                # display_position_status(position, price_precision_digits, amount_precision_digits) # Display potential TSL update
            elif enable_trailing_stop and (last_atr is None or last_atr <= 0):
                 log_warning("Trailing stop enabled but ATR is invalid. Skipping TSL check.")


            # B. Check for Indicator-Based Exits (Optional Secondary Exit)
            # Note: This logic is basic. More sophisticated exits could be added.
            # This acts as a fallback IF SL/TP somehow fail or if you want faster exits.
            execute_indicator_exit = False
            exit_reason = ""
            if position['status'] == 'long' and last_rsi > rsi_overbought:
                 exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} > {rsi_overbought})"
                 execute_indicator_exit = True
            elif position['status'] == 'short' and last_rsi < rsi_oversold:
                 exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} < {rsi_oversold})"
                 execute_indicator_exit = True

            if execute_indicator_exit:
                display_signal("Exit", position['status'], exit_reason)
                log_warning(f"Attempting to close {position['status'].upper()} position via FALLBACK market order due to indicator signal.")

                # IMPORTANT: Cancel existing SL and TP orders *before* sending market order
                sl_id_to_cancel = position.get('sl_order_id')
                tp_id_to_cancel = position.get('tp_order_id')
                orders_cancelled_successfully = True # Assume success initially

                for order_id, order_type in [(sl_id_to_cancel, "SL"), (tp_id_to_cancel, "TP")]:
                    if order_id:
                        log_info(f"Cancelling existing {order_type} order {order_id} before fallback exit...")
                        try:
                            cancel_order_with_retry(exchange, order_id, symbol)
                        except ccxt.OrderNotFound:
                            log_info(f"{order_type} order {order_id} already closed/cancelled.")
                        except Exception as e:
                            log_error(f"Error cancelling {order_type} order {order_id} during fallback exit: {e}", exc_info=False)
                            orders_cancelled_successfully = False # Mark cancellation as potentially failed

                if not orders_cancelled_successfully:
                     log_error("Failed to cancel one or both SL/TP orders. Aborting fallback market exit to avoid potential issues. MANUAL INTERVENTION MAY BE REQUIRED.")
                else:
                    # Proceed with market exit order (use reduceOnly=True)
                    close_side = 'sell' if position['status'] == 'long' else 'buy'
                    exit_qty = position['quantity']
                    if exit_qty is None or exit_qty <= 0:
                         log_error("Cannot place fallback exit order: Invalid quantity in position state.")
                    else:
                        order_result = place_market_order(exchange, symbol, close_side, exit_qty, reduce_only=True)

                        # Check if order likely filled (can be tricky with market orders)
                        # Status 'closed' is ideal, but some exchanges might return 'open' briefly
                        # We reset state assuming it worked, check_position_and_orders will confirm next cycle
                        if order_result and order_result.get('id'):
                            log_info(f"Fallback {position['status']} position close order placed: ID {order_result.get('id', 'N/A')}")
                        else:
                            log_error(f"Fallback market order placement FAILED for {position['status']} position.")
                            # Critical: Bot tried to exit but failed. Manual intervention likely needed.

                        # Reset position state immediately after attempting market exit
                        # regardless of confirmation, as intent was to close.
                        log_info("Resetting local position state after indicator-based market exit attempt.")
                        position.update({
                            'status': None, 'entry_price': None, 'quantity': None, 'order_id': None,
                            'stop_loss': None, 'take_profit': None, 'entry_time': None,
                            'sl_order_id': None, 'tp_order_id': None,
                            'highest_price_since_entry': None, 'lowest_price_since_entry': None,
                            'current_trailing_sl_price': None
                        })
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits)
                        # Exit loop for this cycle after attempting exit
                        neon_sleep_timer(sleep_interval_seconds)
                        continue
            else:
                 # If no indicator exit, just log monitoring status
                 log_info(f"Monitoring {position['status'].upper()} position. Waiting for SL/TP ({position.get('sl_order_id')}/{position.get('tp_order_id')}) or TSL update.")


        # --- Check for NEW ENTRIES (only if not currently in a position) ---
        else: # position['status'] is None
            log_info("No active position. Checking for entry signals...")

            # --- Volume Confirmation Check ---
            volume_confirmed = False
            if entry_volume_confirmation_enabled:
                if current_volume is not None and last_volume_ma is not None and last_volume_ma > 0:
                    if current_volume > (last_volume_ma * entry_volume_multiplier):
                        volume_confirmed = True
                        log_debug(f"Volume confirmed: Current Vol ({current_volume:.2f}) > MA Vol ({last_volume_ma:.2f}) * {entry_volume_multiplier}")
                    else:
                        log_debug(f"Volume NOT confirmed: Current Vol ({current_volume:.2f}), MA Vol ({last_volume_ma:.2f}), Threshold ({last_volume_ma * entry_volume_multiplier:.2f})")
                else:
                    log_debug("Volume confirmation enabled but volume or MA data is missing/invalid.")
            else:
                volume_confirmed = True # Skip check if disabled

            # --- Base Signal Conditions ---
            base_long_signal = last_rsi < rsi_oversold and last_stoch_k < stoch_oversold
            base_short_signal = last_rsi > rsi_overbought and last_stoch_k > stoch_overbought

            # --- OB Price Check ---
            long_ob_price_check = False
            long_ob_reason_part = ""
            if base_long_signal and bullish_ob:
                ob_range = bullish_ob['high'] - bullish_ob['low']
                # Allow entry if price is within OB or up to 10% of OB range above high
                entry_zone_high = bullish_ob['high'] + max(ob_range * 0.10, min_tick * 2) # Add min tick buffer
                entry_zone_low = bullish_ob['low']
                if entry_zone_low <= current_price <= entry_zone_high:
                    long_ob_price_check = True
                    long_ob_reason_part = f"Price near Bullish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                else:
                    log_debug(f"Base Long signal met, but price {current_price:.{price_precision_digits}f} outside Bullish OB entry zone.")
            elif base_long_signal:
                 log_debug("Base Long signal met, but no recent Bullish OB found.")

            short_ob_price_check = False
            short_ob_reason_part = ""
            if base_short_signal and bearish_ob:
                ob_range = bearish_ob['high'] - bearish_ob['low']
                # Allow entry if price is within OB or down to 10% of OB range below low
                entry_zone_low = bearish_ob['low'] - max(ob_range * 0.10, min_tick * 2)
                entry_zone_high = bearish_ob['high']
                if entry_zone_low <= current_price <= entry_zone_high:
                    short_ob_price_check = True
                    short_ob_reason_part = f"Price near Bearish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                else:
                     log_debug(f"Base Short signal met, but price {current_price:.{price_precision_digits}f} outside Bearish OB entry zone.")
            elif base_short_signal:
                  log_debug("Base Short signal met, but no recent Bearish OB found.")


            # --- Combine Conditions for Final Entry Signal ---
            long_entry_condition = base_long_signal and long_ob_price_check and volume_confirmed
            short_entry_condition = base_short_signal and short_ob_price_check and volume_confirmed

            long_reason = ""
            short_reason = ""
            if long_entry_condition:
                 long_reason = (f"RSI ({last_rsi:.1f} < {rsi_oversold}), "
                                f"StochK ({last_stoch_k:.1f} < {stoch_oversold}), "
                                f"{long_ob_reason_part}" +
                                (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))
            elif short_entry_condition:
                 short_reason = (f"RSI ({last_rsi:.1f} > {rsi_overbought}), "
                                 f"StochK ({last_stoch_k:.1f} > {stoch_overbought}), "
                                 f"{short_ob_reason_part}" +
                                 (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))
            # Log reasons for failure if base signal was met but other checks failed
            elif base_long_signal and long_ob_price_check and not volume_confirmed:
                 log_debug("Long OB/Price conditions met, but volume not confirmed.")
            elif base_short_signal and short_ob_price_check and not volume_confirmed:
                 log_debug("Short OB/Price conditions met, but volume not confirmed.")


            # --- Execute Entry ---
            if long_entry_condition:
                display_signal("Entry", "long", long_reason)

                # Calculate SL/TP prices
                stop_loss_price = 0.0
                take_profit_price = 0.0
                if enable_atr_sl_tp:
                    if last_atr is None or last_atr <= 0:
                        log_error("Cannot calculate ATR SL/TP: Invalid ATR value. Skipping LONG entry.")
                        continue # Skip to next cycle
                    stop_loss_price = current_price - (last_atr * atr_sl_multiplier)
                    take_profit_price = current_price + (last_atr * atr_tp_multiplier)
                    log_info(f"Calculated ATR-based SL: {stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP: {take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)")
                else: # Fixed percentage
                    stop_loss_price = current_price * (1 - stop_loss_percentage)
                    take_profit_price = current_price * (1 + take_profit_percentage)
                    log_info(f"Calculated Fixed % SL: {stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage*100:.1f}%), TP: {take_profit_price:.{price_precision_digits}f} ({take_profit_percentage*100:.1f}%)")

                # Adjust SL based on Bullish OB low if it provides tighter stop
                # Ensure bullish_ob exists (should be true if long_entry_condition is met with OB check)
                if bullish_ob and bullish_ob['low'] > stop_loss_price:
                    # Set SL just below the OB low (add buffer)
                    adjusted_sl = bullish_ob['low'] * (1 - 0.0005) # Smaller buffer, e.g., 0.05%
                    # Ensure adjustment doesn't push SL too close or above current price
                    if adjusted_sl < current_price and adjusted_sl > stop_loss_price:
                         stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))
                         log_info(f"Adjusted SL tighter based on Bullish OB low: {stop_loss_price:.{price_precision_digits}f}")
                    else:
                         log_warning(f"Could not adjust SL based on OB low {bullish_ob['low']:.{price_precision_digits}f} as it's too close or invalid.")


                # Final SL/TP validation
                if stop_loss_price >= current_price or take_profit_price <= current_price:
                     log_error(f"Invalid SL/TP calculated: SL {stop_loss_price:.{price_precision_digits}f}, TP {take_profit_price:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping LONG entry.")
                     continue

                # Calculate position size
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)
                if quantity is None or quantity <= 0:
                    log_error("Failed to calculate valid position size. Skipping LONG entry.")
                else:
                    # Place entry market order
                    entry_order_result = place_market_order(exchange, symbol, 'buy', quantity, reduce_only=False) # Entry is not reduceOnly
                    if entry_order_result and entry_order_result.get('status') == 'closed': # Check fill status
                        entry_price_actual = entry_order_result.get('average', current_price) # Use filled price if available
                        filled_quantity = entry_order_result.get('filled', quantity) # Use filled qty if available

                        log_info(f"Long position entry order filled: ID {entry_order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f} for {filled_quantity:.{amount_precision_digits}f}")

                        # Place SL/TP orders *after* confirming entry
                        sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'long', filled_quantity, stop_loss_price, take_profit_price)

                        # Update position state ONLY if entry was successful
                        position.update({
                            'status': 'long',
                            'entry_price': entry_price_actual,
                            'quantity': filled_quantity,
                            'order_id': entry_order_result.get('id'),
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'entry_time': pd.Timestamp.now(tz='UTC'),
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None,
                            # Initialize TSL fields
                            'highest_price_since_entry': entry_price_actual,
                            'lowest_price_since_entry': None,
                            'current_trailing_sl_price': None
                        })
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits) # Show new status

                        if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):
                            log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                    else:
                        log_error("Failed to place or confirm fill for long entry order.")

            elif short_entry_condition: # Use elif to prevent long/short in same cycle
                display_signal("Entry", "short", short_reason)

                # Calculate SL/TP prices
                stop_loss_price = 0.0
                take_profit_price = 0.0
                if enable_atr_sl_tp:
                    if last_atr is None or last_atr <= 0:
                        log_error("Cannot calculate ATR SL/TP: Invalid ATR value. Skipping SHORT entry.")
                        continue
                    stop_loss_price = current_price + (last_atr * atr_sl_multiplier)
                    take_profit_price = current_price - (last_atr * atr_tp_multiplier)
                    log_info(f"Calculated ATR-based SL: {stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP: {take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)")
                else: # Fixed percentage
                    stop_loss_price = current_price * (1 + stop_loss_percentage)
                    take_profit_price = current_price * (1 - take_profit_percentage)
                    log_info(f"Calculated Fixed % SL: {stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage*100:.1f}%), TP: {take_profit_price:.{price_precision_digits}f} ({take_profit_percentage*100:.1f}%)")

                # Adjust SL based on Bearish OB high if it provides tighter stop
                if bearish_ob and bearish_ob['high'] < stop_loss_price:
                     adjusted_sl = bearish_ob['high'] * (1 + 0.0005) # Small buffer above high
                     if adjusted_sl > current_price and adjusted_sl < stop_loss_price:
                         stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))
                         log_info(f"Adjusted SL tighter based on Bearish OB high: {stop_loss_price:.{price_precision_digits}f}")
                     else:
                         log_warning(f"Could not adjust SL based on OB high {bearish_ob['high']:.{price_precision_digits}f} as it's too close or invalid.")

                # Final SL/TP validation
                if stop_loss_price <= current_price or take_profit_price >= current_price:
                     log_error(f"Invalid SL/TP calculated: SL {stop_loss_price:.{price_precision_digits}f}, TP {take_profit_price:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping SHORT entry.")
                     continue

                # Calculate position size
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)
                if quantity is None or quantity <= 0:
                    log_error("Failed to calculate valid position size. Skipping SHORT entry.")
                else:
                    # Place entry market order
                    entry_order_result = place_market_order(exchange, symbol, 'sell', quantity, reduce_only=False)
                    if entry_order_result and entry_order_result.get('status') == 'closed':
                        entry_price_actual = entry_order_result.get('average', current_price)
                        filled_quantity = entry_order_result.get('filled', quantity)

                        log_info(f"Short position entry order filled: ID {entry_order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f} for {filled_quantity:.{amount_precision_digits}f}")

                        # Place SL/TP orders
                        sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'short', filled_quantity, stop_loss_price, take_profit_price)

                        # Update position state
                        position.update({
                            'status': 'short',
                            'entry_price': entry_price_actual,
                            'quantity': filled_quantity,
                            'order_id': entry_order_result.get('id'),
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'entry_time': pd.Timestamp.now(tz='UTC'),
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None,
                            # Initialize TSL fields
                            'highest_price_since_entry': None,
                            'lowest_price_since_entry': entry_price_actual,
                            'current_trailing_sl_price': None
                        })
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits)

                        if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):
                            log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                    else:
                        log_error("Failed to place or confirm fill for short entry order.")

            else: # No entry condition met
                log_info("Conditions not met for new entry.")


        # 7. Wait for the next cycle
        log_info(f"Cycle complete. Waiting for {sleep_interval_seconds} seconds...")
        neon_sleep_timer(sleep_interval_seconds)

    # --- Graceful Shutdown Handling ---
    except KeyboardInterrupt:
        log_info("Keyboard interrupt detected (Ctrl+C). Stopping the bot...")
        save_position_state()  # Save final state before exiting

        # Attempt to cancel open orders for the symbol
        if not SIMULATION_MODE:
            log_info(f"Attempting to cancel all open orders for {symbol}...")
            try:
                # Fetch open orders first (use retry)
                @api_retry_decorator
                def fetch_open_orders_on_exit(exch, sym):
                    return exch.fetch_open_orders(sym)

                open_orders = fetch_open_orders_on_exit(exchange, symbol)

                if not open_orders:
                    log_info("No open orders found to cancel.")
                else:
                    log_warning(f"Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...")
                    cancelled_count = 0
                    failed_count = 0
                    for order in open_orders:
                        order_id = order.get('id')
                        if not order_id: continue # Skip if no ID
                        try:
                            log_info(f"Cancelling order ID: {order_id} (Type: {order.get('type', 'N/A')}, Side: {order.get('side','N/A')})...")
                            # Use cancel helper with retry
                            cancel_order_with_retry(exchange, order_id, symbol)
                            cancelled_count += 1
                            time.sleep(0.3) # Small delay between cancellations
                        except ccxt.OrderNotFound:
                             log_info(f"Order {order_id} already closed/cancelled.")
                             cancelled_count += 1 # Count as effectively cancelled
                        except Exception as cancel_e:
                            # Error logged by helper
                            log_error(f"Failed to cancel order {order_id} on exit after retries.", exc_info=False)
                            failed_count +=1
                    log_info(f"Order cancellation attempt complete. Success/Already Closed: {cancelled_count}, Failed: {failed_count}")
            except Exception as e:
                log_error(f"Error fetching or cancelling open orders on exit: {e}", exc_info=True)
        else:
             log_info("Simulation mode: Skipping order cancellation on exit.")

        break # Exit the main while loop

    # --- Robust Error Handling for the Main Loop ---
    except ccxt.RateLimitExceeded as e:
        log_error(f"Main loop Rate Limit Error: {e}. Waiting longer (default + 60s)...", exc_info=False)
        neon_sleep_timer(sleep_interval_seconds + 60) # Wait longer
    except ccxt.NetworkError as e:
        log_error(f"Main loop Network Error: {e}. Default sleep + retry...", exc_info=False)
        neon_sleep_timer(sleep_interval_seconds) # Rely on retry decorator for backoff
    except ccxt.ExchangeError as e: # Catch other exchange-specific issues
        log_error(f"Main loop Exchange Error: {e}. Default sleep + retry...", exc_info=False)
        # Could add specific handling for maintenance errors etc. here
        neon_sleep_timer(sleep_interval_seconds)
    except Exception as e:
        log_error(f"CRITICAL unexpected error in main loop: {e}", exc_info=True)
        log_info("Attempting to recover by saving state and waiting 60s before next cycle...")
        try:
            save_position_state() # Save state on critical error
        except Exception as save_e:
            log_error(f"Failed to save state during critical error handling: {save_e}", exc_info=True)
        neon_sleep_timer(60) # Wait before trying next cycle

# --- Bot Exit ---
print_shutdown_message()
sys.exit(0)
```
````python

--- Add near configuration loading ---                               ENABLE_FIB_PIVOT_ACTIONS: bool = config.get("enable_fib_pivot_actions", False) # Master switch for Fib pivots                                 PIVOT_TIMEFRAME: str = config.get("pivot_timeframe", "1d") # Timeframe for Pivot calculation (e.g., '1d', '1W')
FIB_LEVELS_TO_CALC: List[float] = config.get("fib_levels_to_calc", [0.382, 0.618, 1.0]) # Fibonacci levels (based on H-L range)
FIB_NEAREST_COUNT: int = int(config.get("fib_nearest_count", 5)) # How many nearest levels to track
FIB_ENTRY_CONFIRM_PERCENT: float = float(config.get("fib_entry_confirm_percent", 0.002)) # Price must be within X% of a Fib support(long)/resistance(short)
FIB_EXIT_WARN_PERCENT: float = float(config.get("fib_exit_warn_percent", 0.0015)) # Warn/Exit if price within Y% of Fib resistance(long)/support(short)
FIB_EXIT_ACTION: str = config.get("fib_exit_action", "warn").lower() # Action on exit warning: "warn", "exit"

if ENABLE_FIB_PIVOT_ACTIONS:
    log_info(f"Fibonacci Pivot Actions: ENABLED (Pivot TF: {PIVOT_TIMEFRAME}, Levels: {FIB_LEVELS_TO_CALC}, "                                              f"Nearest: {FIB_NEAREST_COUNT}, Entry%: {FIB_ENTRY_CONFIRM_PERCENT*100:.3f}%, "
             f"Exit%: {FIB_EXIT_WARN_PERCENT*100:.3f}%, Exit Action: {FIB_EXIT_ACTION.upper()})")
    if FIB_NEAREST_COUNT <= 0 or FIB_ENTRY_CONFIRM_PERCENT <= 0 or FIB_EXIT_WARN_PERCENT <= 0:
        log_error("CRITICAL: Invalid Fibonacci Pivot parameter(s) in config. Values must be positive.")                                               sys.exit(1)
    if PIVOT_TIMEFRAME not in exchange.timeframes:
        log_error(f"CRITICAL: Pivot timeframe '{PIVOT_TIMEFRAME}' not supported by exchange {exchange_id}. Available: {list(exchange.timeframes.keys())}")
        sys.exit(1)
    if FIB_EXIT_ACTION not in ["warn", "exit"]:                                log_error("CRITICAL: Invalid 'fib_exit_action' in config. Use 'warn' or 'exit'.")                                                             sys.exit(1)
else:
    log_info("Fibonacci Pivot Actions: DISABLED")                      

# --- Helper Functions for Pivots ---                                  
def calculate_fibonacci_pivots(high: float, low: float, close: float) -> Dict[str, float]:                                                        """
    Calculates standard Pivot Points and Fibonacci Pivot Levels based on HLC.
    """                                                                    if high <= low: # Basic validation
        return {}
                                                                           pivot_levels = {}
    pivot_range = high - low

    # Standard Pivot Point (PP)                                            pp = (high + low + close) / 3
    pivot_levels['PP'] = pp                                            
    # Standard Resistance Levels
    pivot_levels['R1'] = (2 * pp) - low
    pivot_levels['R2'] = pp + pivot_range                                  pivot_levels['R3'] = high + 2 * (pp - low)

    # Standard Support Levels
    pivot_levels['S1'] = (2 * pp) - high
    pivot_levels['S2'] = pp - pivot_range                                  pivot_levels['S3'] = low - 2 * (high - pp)

    # Fibonacci Levels based on PP and Range
    for level in FIB_LEVELS_TO_CALC:                                           pivot_levels[f'FR{level:.3f}'] = pp + (level * pivot_range)
        pivot_levels[f'FS{level:.3f}'] = pp - (level * pivot_range)    
    # Ensure levels are floats and remove potential duplicates (though unlikely with names)
    unique_levels = {name: float(price) for name, price in pivot_levels.items() if isinstance(price, (int, float))}                               return unique_levels

def get_nearest_fib_levels(all_levels: Dict[str, float], current_price: float, count: int) -> Dict[str, float]:
    """
    Finds the N nearest pivot levels to the current price.                 Returns a dictionary of {level_name: price}.
    """
    if not all_levels or count <= 0:
        return {}

    # Calculate absolute distance for each level
    distances = {name: abs(price - current_price) for name, price in all_levels.items()}                                                      
    # Sort levels by distance                                              sorted_levels = sorted(distances.items(), key=lambda item: item[1])
                                                                           # Get the top N nearest levels
    nearest_levels = {name: all_levels[name] for name, dist in sorted_levels[:count]}
                                                                           return nearest_levels


# --- Add to main trading loop ---

# Inside the `while True:` loop, AFTER fetching base timeframe data and calculating indicators,
# but BEFORE checking position/entry/exit logic:

        # --- Calculate Fibonacci Pivot Levels (if enabled) ---                nearest_pivots: Dict[str, float] = {}
        pivot_support_levels: Dict[str, float] = {}
        pivot_resistance_levels: Dict[str, float] = {}
        if ENABLE_FIB_PIVOT_ACTIONS:
            log_debug(f"Calculating Fibonacci Pivots based on {PIVOT_TIMEFRAME} timeframe...")
            try:
                # Fetch data for the pivot timeframe (need previous closed candle)
                # Fetch 2 candles to ensure we have the completed previous one
                pivot_ohlcv_df = fetch_ohlcv_data(exchange, symbol, PIVOT_TIMEFRAME, limit_count=2)
                                                                                       if pivot_ohlcv_df is not None and len(pivot_ohlcv_df) >= 2:                                                                                       # Use the second to last row (index -2) which is the *last completed* candle
                    prev_pivot_candle = pivot_ohlcv_df.iloc[-2]                            prev_high = float(prev_pivot_candle['high'])
                    prev_low = float(prev_pivot_candle['low'])
                    prev_close = float(prev_pivot_candle['close'])
                    log_debug(f"Previous {PIVOT_TIMEFRAME} candle HLC: H={prev_high:.{price_precision_digits}f}, L={prev_low:.{price_precision_digits}f}, C={prev_close:.{price_precision_digits}f}")

                    all_calculated_pivots = calculate_fibonacci_pivots(prev_high, prev_low, prev_close)

                    if all_calculated_pivots:                                                  nearest_pivots = get_nearest_fib_levels(all_calculated_pivots, current_price, FIB_NEAREST_COUNT)
                        # Separate into support and resistance relative to current price                                                                              pivot_support_levels = {name: price for name, price in nearest_pivots.items() if price < current_price}                                       pivot_resistance_levels = {name: price for name, price in nearest_pivots.items() if price >= current_price} # Include levels at current price as resistance

                        log_info(f"Nearest {len(nearest_pivots)} Pivots: " + ", ".join([f"{n}={p:.{price_precision_digits}f}" for n, p in sorted(nearest_pivots.items(), key=lambda item: item[1])]))                                        log_debug(f"Nearest Support Pivots: {pivot_support_levels}")                                                                                  log_debug(f"Nearest Resistance Pivots: {pivot_resistance_levels}")
                    else:                                                                      log_warning(f"Failed to calculate pivot levels from {PIVOT_TIMEFRAME} data.")                                                         else:
                    log_warning(f"Could not fetch sufficient data ({len(pivot_ohlcv_df) if pivot_ohlcv_df is not None else 0} candles) for {PIVOT_TIMEFRAME} pivots.")
            except Exception as pivot_e:
                 log_error(f"Error during Fibonacci Pivot calculation: {pivot_e}", exc_info=True)


        # --- Modify Entry Logic ---
        # Inside the `else: # position['status'] is None` block:       
            # --- Combine Conditions for Final Entry Signal ---                    fib_long_confirm = False
            fib_short_confirm = False                                              fib_reason_part = ""
                                                                                   if ENABLE_FIB_PIVOT_ACTIONS and not nearest_pivots:
                log_warning("Fib Pivot confirmation enabled, but no pivot levels calculated. Entry check will fail.")

            if ENABLE_FIB_PIVOT_ACTIONS and nearest_pivots:
                # Check Long confirmation against nearest SUPPORT pivots
                if base_long_signal:                                                       for name, price in pivot_support_levels.items():
                        if abs(current_price - price) / price <= FIB_ENTRY_CONFIRM_PERCENT:
                            fib_long_confirm = True
                            fib_reason_part = f"Near Fib Support {name}={price:.{price_precision_digits}f} ({FIB_ENTRY_CONFIRM_PERCENT*100:.3f}%)"                                                                                               break # Found one confirmation
                    if not fib_long_confirm:                                                   log_debug(f"Base Long signal, but price {current_price:.{price_precision_digits}f} not near any Fib support level within {FIB_ENTRY_CONFIRM_PERCENT*100:.3f}%.")

                # Check Short confirmation against nearest RESISTANCE pivots
                if base_short_signal:
                    for name, price in pivot_resistance_levels.items():
                        # Ensure price > 0 for division                                        if price > 0 and abs(current_price - price) / price <= FIB_ENTRY_CONFIRM_PERCENT:
                            fib_short_confirm = True
                            fib_reason_part = f"Near Fib Resistance {name}={price:.{price_precision_digits}f} ({FIB_ENTRY_CONFIRM_PERCENT*100:.3f}%)"
                            break # Found one confirmation
                    if not fib_short_confirm:
                        log_debug(f"Base Short signal, but price {current_price:.{price_precision_digits}f} not near any Fib resistance level within {FIB_ENTRY_CONFIRM_PERCENT*100:.3f}%.")
            else:
                 # If Fib pivots are disabled, confirmation is implicitly true                                                                                 fib_long_confirm = True
                 fib_short_confirm = True


            # --- Update Final Entry Conditions ---
            long_entry_condition = base_long_signal and long_ob_price_check and volume_confirmed and fib_long_confirm
            short_entry_condition = base_short_signal and short_ob_price_check and volume_confirmed and fib_short_confirm
                                                                                   long_reason = ""
            short_reason = ""
            if long_entry_condition:
                 long_reason = (f"RSI ({last_rsi:.1f} < {rsi_oversold}), "
                                f"StochK ({last_stoch_k:.1f} < {stoch_oversold}), "                                                                                           f"{long_ob_reason_part}" +
                                (f", {fib_reason_part}" if ENABLE_FIB_PIVOT_ACTIONS and fib_reason_part else "") +
                                (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))                                                      elif short_entry_condition:
                 short_reason = (f"RSI ({last_rsi:.1f} > {rsi_overbought}), "
                                 f"StochK ({last_stoch_k:.1f} > {stoch_overbought}), "                                                                                         f"{short_ob_reason_part}" +
                                 (f", {fib_reason_part}" if ENABLE_FIB_PIVOT_ACTIONS and fib_reason_part else "") +
                                 (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))                                                     # Log reasons for failure if base signal was met but other checks failed
            elif base_long_signal and long_ob_price_check and volume_confirmed and not fib_long_confirm and ENABLE_FIB_PIVOT_ACTIONS:
                 log_debug("Long Base/OB/Vol conditions met, but Fib support confirmation failed.")
            elif base_short_signal and short_ob_price_check and volume_confirmed and not fib_short_confirm and ENABLE_FIB_PIVOT_ACTIONS:
                 log_debug("Short Base/OB/Vol conditions met, but Fib resistance confirmation failed.")
            # ... (other existing failure logs) ...


        # --- Add Fib Pivot Exit Check ---
        # Inside the `if position['status'] is not None:` block, AFTER TSL check, BEFORE indicator exit check:

            # C. Check for Fibonacci Pivot Exit Warning/Action (if enabled)
            fib_exit_triggered = False                                             fib_exit_reason = ""
            if ENABLE_FIB_PIVOT_ACTIONS and nearest_pivots:                            if position['status'] == 'long':
                    # Check proximity to nearest RESISTANCE levels                         for name, price in pivot_resistance_levels.items():
                         # Ensure price > 0 for division
                         if price > 0 and abs(current_price - price) / price <= FIB_EXIT_WARN_PERCENT:                                                                    fib_exit_triggered = True
                            fib_exit_reason = f"Price {current_price:.{price_precision_digits}f} approaching Fib Resistance {name}={price:.{price_precision_digits}f} ({FIB_EXIT_WARN_PERCENT*100:.3f}%)"
                            break # Found one resistance level too close
                elif position['status'] == 'short':
                    # Check proximity to nearest SUPPORT levels
                    for name, price in pivot_support_levels.items():
                         # Ensure price > 0 for division
                         if price > 0 and abs(current_price - price) / price <= FIB_EXIT_WARN_PERCENT:
                            fib_exit_triggered = True
                            fib_exit_reason = f"Price {current_price:.{price_precision_digits}f} approaching Fib Support {name}={price:.{price_precision_digits}f} ({FIB_EXIT_WARN_PERCENT*100:.3f}%)"
                            break # Found one support level too close

            if fib_exit_triggered:
                if FIB_EXIT_ACTION == "warn":                                               log_warning(f"{NEON_YELLOW}Fib Pivot Exit Warning: {fib_exit_reason}{RESET}")
                     # Action: Could potentially tighten TSL here if desired, but currently just warns.
                elif FIB_EXIT_ACTION == "exit":
                     log_warning(f"{NEON_RED}Fib Pivot Exit Signal: {fib_exit_reason}. Attempting market exit.{RESET}")                                            display_signal("Exit (Fib Pivot)", position['status'], fib_exit_reason)                                                  
                     # --- Execute Market Exit (similar to indicator exit logic) ---
                     log_warning(f"Attempting to close {position['status'].upper()} position via market order due to Fib Pivot proximity.")
                     sl_id_to_cancel = position.get('sl_order_id')
                     tp_id_to_cancel = position.get('tp_order_id')
                     orders_cancelled_successfully = True

                     for order_id, order_type in [(sl_id_to_cancel, "SL"), (tp_id_to_cancel, "TP")]:
                         if order_id:                                                               log_info(f"Cancelling existing {order_type} order {order_id} before Fib Pivot exit...")                                                       try: cancel_order_with_retry(exchange, order_id, symbol)
                             except ccxt.OrderNotFound: log_info(f"{order_type} order {order_id} already closed/cancelled.")
                             except Exception as e:
                                 log_error(f"Error cancelling {order_type} order {order_id} during Fib Pivot exit: {e}", exc_info=False)
                                 orders_cancelled_successfully = False                                                                                             if not orders_cancelled_successfully:
                         log_error("Failed to cancel SL/TP orders. Aborting Fib Pivot market exit.")
                     else:
                         close_side = 'sell' if position['status'] == 'long' else 'buy'
                         exit_qty = position.get('quantity')
                         if exit_qty is None or exit_qty <= 0:                                      log_error("Cannot place Fib Pivot exit order: Invalid quantity in position state.")
                         else:
                             order_result = place_market_order(exchange, symbol, close_side, exit_qty, reduce_only=True)
                             if order_result and order_result.get('id'):
                                 log_info(f"Fib Pivot-based {position['status']} position close order placed: ID {order_result.get('id', 'N/A')}")
                             else:
                                 log_error(f"Fib Pivot-based market exit order FAILED for {position['status']} position.")

                             # Reset state regardless of order confirmation (intent was to close)
                             log_info("Resetting local position state after Fib Pivot-based market exit attempt.")
                             default_state = {k: None for k in position.keys()}                                                                                            position.update(default_state)
                             save_position_state()
                             display_position_status(position, price_precision_digits, amount_precision_digits)                                                            # Go to next cycle immediately after exit attempt
                             neon_sleep_timer(sleep_interval_seconds)
                             continue # Skip remaining checks in this cycle

            # D. Check for Indicator-Based Exits (Only if Fib exit didn't happen)
            if not fib_exit_triggered or FIB_EXIT_ACTION == "warn": # Proceed if no Fib exit or only warned
                execute_indicator_exit = False
                exit_reason = ""                                                       # ... (rest of the existing indicator exit logic remains the same) ...                                                                        if position['status'] == 'long' and last_rsi > rsi_overbought:
                    exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} > {rsi_overbought})"
                    execute_indicator_exit = True
                elif position['status'] == 'short' and last_rsi < rsi_oversold:
                    exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} < {rsi_oversold})"                                                                  execute_indicator_exit = True

                if execute_indicator_exit:
                    # ... (Existing logic to cancel SL/TP and place market exit order) ...
                    # Ensure this logic also resets the position state correctly.
                    pass # Placeholder for existing indicator exit logic
                else:                                                                      # If no exit, log monitoring status
                    log_info(f"Monitoring {position['status'].upper()} position. Waiting for SL/TP ({position.get('sl_order_id') or 'N/A'}/{position.get('tp_order_id') or 'N/A'}) or TSL update.")

        # --- End of main loop ---
```
# --- Add near configuration loading ---
ENABLE_OB_PROFIT_ACTIONS: bool = config.get("enable_ob_profit_actions", False) # Master switch
OB_FETCH_DEPTH: int = int(config.get("ob_fetch_depth", 20)) # How many levels of bids/asks to fetch
# Threshold for wall detection: Size in quote currency (e.g., USD) relative to current position value
OB_WALL_SIZE_MULT: float = float(config.get("ob_wall_size_mult", 5.0)) # e.g., wall > 5x current position value
# How close the wall needs to be to current price (percentage)
OB_WALL_DISTANCE_PERCENT: float = float(config.get("ob_wall_distance_percent", 0.005)) # e.g., within 0.5%
# How much to add when scaling in (multiplier of ORIGINAL entry quantity)
OB_SCALE_IN_AMOUNT_MULT: float = float(config.get("ob_scale_in_amount_mult", 0.5)) # e.g., add 50% of original size
OB_MAX_SCALE_INS: int = int(config.get("ob_max_scale_ins", 2)) # Max number of times to scale into a position

if ENABLE_OB_PROFIT_ACTIONS:
    log_info(f"Order Book Profit Actions: ENABLED (Depth: {OB_FETCH_DEPTH}, Wall Size: {OB_WALL_SIZE_MULT}x PosVal, "
             f"Distance: {OB_WALL_DISTANCE_PERCENT*100:.2f}%, Scale Mult: {OB_SCALE_IN_AMOUNT_MULT}, Max Scales: {OB_MAX_SCALE_INS})")
    if OB_FETCH_DEPTH <= 0 or OB_WALL_SIZE_MULT <= 0 or OB_WALL_DISTANCE_PERCENT <= 0 or OB_SCALE_IN_AMOUNT_MULT <= 0 or OB_MAX_SCALE_INS < 0:
        log_error("CRITICAL: Invalid order book analysis parameter(s) in config. Values must be positive (Max Scales >= 0).")
        sys.exit(1)
    # Add check if exchange supports fetching order book
    if not exchange.has.get('fetchL2OrderBook'):
         log_error(f"CRITICAL: Exchange {exchange_id} does not support fetchL2OrderBook via CCXT. Disable 'enable_ob_profit_actions'.")
         sys.exit(1)
else:
    log_info("Order Book Profit Actions: DISABLED")


# --- Add to Position Management State ---
position: Dict[str, Any] = {
    # ... (existing fields) ...
    'original_quantity': None, # Float: Store the quantity of the initial entry
    'times_scaled_in': 0,      # Int: Counter for scale-in operations
    'last_ob_action_time': None # pd.Timestamp: Prevent rapid consecutive OB actions
}
# --- Ensure load_position_state handles these new fields ---
# (Modify load_position_state to initialize these keys if missing in the file)
# Example addition inside load_position_state, after loading `loaded_state`:
#   ...
#   position['original_quantity'] = loaded_state.get('original_quantity', position.get('quantity')) # Initialize intelligently
#   position['times_scaled_in'] = loaded_state.get('times_scaled_in', 0)
#   last_ob_action_time_str = loaded_state.get('last_ob_action_time')
#   if last_ob_action_time_str:
#       try:
#           ts_ob = pd.Timestamp(last_ob_action_time_str)
#           if ts_ob.tzinfo is None: ts_ob = ts_ob.tz_localize('UTC')
#           else: ts_ob = ts_ob.tz_convert('UTC')
#           position['last_ob_action_time'] = ts_ob
#       except ValueError:
#           position['last_ob_action_time'] = None
#   else:
#       position['last_ob_action_time'] = None
#   ...

# --- Add new Order Book Analysis Function (Decorated) ---
@api_retry_decorator
def fetch_and_analyze_order_book(exchange_instance: ccxt.Exchange, trading_symbol: str, current_pos_value_quote: float) -> Tuple[bool, bool, float, float]:
    """
    Fetches L2 order book and analyzes for significant walls near the current price.
    Returns: (large_support_found, large_resistance_found, total_support_value, total_resistance_value)
    """
    log_debug(f"Fetching order book for {trading_symbol} (Depth: {OB_FETCH_DEPTH})...")
    try:
        order_book = exchange_instance.fetch_l2_order_book(trading_symbol, limit=OB_FETCH_DEPTH)

        if not order_book or not order_book.get('bids') or not order_book.get('asks'):
            log_warning("Order book data is empty or incomplete.")
            return False, False, 0.0, 0.0

        ticker = exchange_instance.fetch_ticker(trading_symbol) # Get current price again for accuracy
        if not ticker or 'last' not in ticker:
             log_warning("Could not fetch ticker for current price during OB analysis.")
             return False, False, 0.0, 0.0
        current_price = ticker['last']

        price_distance_threshold = current_price * OB_WALL_DISTANCE_PERCENT
        wall_value_threshold = current_pos_value_quote * OB_WALL_SIZE_MULT

        log_debug(f"OB Analysis | Current Price: {current_price:.{price_precision_digits}f}, "
                  f"Distance Threshold: +/- {price_distance_threshold:.{price_precision_digits}f} ({OB_WALL_DISTANCE_PERCENT*100:.2f}%), "
                  f"Wall Value Threshold: {wall_value_threshold:.{price_precision_digits}f} {market_info.get('quote', '')}")

        # Analyze Bids (Support)
        total_support_value_near = 0.0
        large_support_found = False
        for price, amount in order_book['bids']:
            if price >= current_price - price_distance_threshold:
                level_value = price * amount
                total_support_value_near += level_value
                if level_value >= wall_value_threshold:
                    large_support_found = True
                    log_debug(f"Large Support Wall Found: Price={price:.{price_precision_digits}f}, Amount={amount:.{amount_precision_digits}f}, Value={level_value:.2f}")
                    # break # Found one, no need to check further bids for the flag (but continue summing total)
            else:
                break # Bids are sorted descending, no need to check further

        # Analyze Asks (Resistance)
        total_resistance_value_near = 0.0
        large_resistance_found = False
        for price, amount in order_book['asks']:
            if price <= current_price + price_distance_threshold:
                level_value = price * amount
                total_resistance_value_near += level_value
                if level_value >= wall_value_threshold:
                    large_resistance_found = True
                    log_debug(f"Large Resistance Wall Found: Price={price:.{price_precision_digits}f}, Amount={amount:.{amount_precision_digits}f}, Value={level_value:.2f}")
                    # break # Found one, no need to check further asks for the flag (but continue summing total)
            else:
                break # Asks are sorted ascending, no need to check further

        log_info(f"OB Analysis Result | Support Found: {large_support_found}, Resistance Found: {large_resistance_found} "
                 f"| Near Support Value: {total_support_value_near:.2f}, Near Resistance Value: {total_resistance_value_near:.2f}")

        return large_support_found, large_resistance_found, total_support_value_near, total_resistance_value_near

    except ccxt.NetworkError as e:
         # Handled by decorator, but log specific context
         log_warning(f"Network error fetching order book: {e}")
         raise # Re-raise for decorator
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error fetching/analyzing order book: {e}")
        return False, False, 0.0, 0.0 # Treat as no walls found on error
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected error fetching/analyzing order book: {e}", exc_info=True)
        return False, False, 0.0, 0.0 # Treat as no walls found on error


# --- Modify the main trading loop ---
# Inside the `while True:` loop, within the `if position['status'] is not None:` block:

        # --- Check if ALREADY IN A POSITION ---
        if position['status'] is not None:
            log_info(f"Currently in {position['status'].upper()} position.")
            # Ensure entry price and quantity are valid before proceeding
            if position.get('entry_price') is None or position.get('quantity') is None or position['quantity'] <= 0:
                 log_error("Position state has invalid entry_price or quantity. Cannot manage position.")
                 # Consider resetting state here or attempting recovery
                 neon_sleep_timer(sleep_interval_seconds)
                 continue

            # A. Check Trailing Stop Logic (if enabled)
            update_trailing_stop(exchange, symbol, current_price, last_atr if last_atr else 0.0)

            # B. Check Order Book Actions (if enabled and profitable)
            ob_action_taken_this_cycle = False # Flag to prevent other actions if OB action occurs
            if ENABLE_OB_PROFIT_ACTIONS:                                               is_profitable = (position['status'] == 'long' and current_price > position['entry_price']) or \
                                (position['status'] == 'short' and current_price < position['entry_price'])

                # Add time check to prevent rapid actions
                can_take_ob_action = True
                if position.get('last_ob_action_time'):
                    time_since_last_action = pd.Timestamp.now(tz='UTC') - position['last_ob_action_time']
                    # Prevent action if last OB action was within, say, half the sleep interval
                    if time_since_last_action < pd.Timedelta(seconds=sleep_interval_seconds / 2):
                        log_debug(f"Skipping OB check: Too soon since last OB action ({time_since_last_action.total_seconds():.0f}s ago).")
                        can_take_ob_action = False

                if is_profitable and can_take_ob_action:
                    log_info("Position is profitable. Analyzing order book for potential actions...")
                    current_position_value_quote = position['quantity'] * current_price
                                                                                           try:
                        support_found, resistance_found, _, _ = fetch_and_analyze_order_book(
                            exchange, symbol, current_position_value_quote
                        )
                                                                                               # --- OB Early Exit Logic ---
                        if position['status'] == 'long' and resistance_found:
                            log_warning(f"{NEON_RED}Large resistance wall detected ahead! Attempting early exit from LONG position.{RESET}")
                            ob_action_taken_this_cycle = True
                            # --- Execute Market Exit ---
                            log_warning(f"Attempting to close LONG position via market order due to OB resistance.")
                            sl_id_to_cancel = position.get('sl_order_id')                                                                                                 tp_id_to_cancel = position.get('tp_order_id')
                            orders_cancelled_successfully = True
                            for order_id, order_type in [(sl_id_to_cancel, "SL"), (tp_id_to_cancel, "TP")]:
                                if order_id:
                                    log_info(f"Cancelling existing {order_type} order {order_id} before OB exit...")
                                    try: cancel_order_with_retry(exchange, order_id, symbol)
                                    except ccxt.OrderNotFound: log_info(f"{order_type} order {order_id} already closed/cancelled.")
                                    except Exception as e:
                                        log_error(f"Error cancelling {order_type} order {order_id} during OB exit: {e}", exc_info=False)
                                        orders_cancelled_successfully = False
                            if not orders_cancelled_successfully:
                                log_error("Failed to cancel SL/TP orders. Aborting OB market exit.")                                                                      else:
                                exit_qty = position.get('quantity')                                    if exit_qty and exit_qty > 0:
                                    order_result = place_market_order(exchange, symbol, 'sell', exit_qty, reduce_only=True)                                                       if order_result and order_result.get('id'):
                                        log_info(f"OB-based LONG position close order placed: ID {order_result.get('id', 'N/A')}")
                                        position['last_ob_action_time'] = pd.Timestamp.now(tz='UTC') # Record action time
                                    else:                                                                      log_error("OB-based market exit order FAILED.")
                                    # Reset state regardless of order confirmation (intent was to close)                                                                          log_info("Resetting local position state after OB-based market exit attempt.")
                                    default_state = {k: None for k in position.keys()}
                                    position.update(default_state)
                                    save_position_state()                                              else: log_error("Cannot place OB exit order: Invalid quantity in position state.")
                            # --- End Market Exit ---

                        elif position['status'] == 'short' and support_found:                                                                                             log_warning(f"{NEON_GREEN}Large support wall detected below! Attempting early exit from SHORT position.{RESET}")
                            ob_action_taken_this_cycle = True
                            # --- Execute Market Exit ---
                            log_warning(f"Attempting to close SHORT position via market order due to OB support.")
                            sl_id_to_cancel = position.get('sl_order_id')
                            tp_id_to_cancel = position.get('tp_order_id')
                            orders_cancelled_successfully = True
                            for order_id, order_type in [(sl_id_to_cancel, "SL"), (tp_id_to_cancel, "TP")]:
                                if order_id:
                                    log_info(f"Cancelling existing {order_type} order {order_id} before OB exit...")
                                    try: cancel_order_with_retry(exchange, order_id, symbol)
                                    except ccxt.OrderNotFound: log_info(f"{order_type} order {order_id} already closed/cancelled.")
                                    except Exception as e:                                                     log_error(f"Error cancelling {order_type} order {order_id} during OB exit: {e}", exc_info=False)
                                        orders_cancelled_successfully = False
                            if not orders_cancelled_successfully:
                                log_error("Failed to cancel SL/TP orders. Aborting OB market exit.")
                            else:
                                exit_qty = position.get('quantity')
                                if exit_qty and exit_qty > 0:
                                    order_result = place_market_order(exchange, symbol, 'buy', exit_qty, reduce_only=True)
                                    if order_result and order_result.get('id'):
                                        log_info(f"OB-based SHORT position close order placed: ID {order_result.get('id', 'N/A')}")
                                        position['last_ob_action_time'] = pd.Timestamp.now(tz='UTC') # Record action time
                                    else:
                                        log_error("OB-based market exit order FAILED.")
                                    # Reset state regardless of order confirmation (intent was to close)
                                    log_info("Resetting local position state after OB-based market exit attempt.")
                                    default_state = {k: None for k in position.keys()}                                                                                            position.update(default_state)
                                    save_position_state()
                                else: log_error("Cannot place OB exit order: Invalid quantity in position state.")
                            # --- End Market Exit ---

                        # --- OB Scale-In Logic (only if no exit occurred and max scale-ins not reached) ---
                        elif not ob_action_taken_this_cycle and position.get('times_scaled_in', 0) < OB_MAX_SCALE_INS:
                            scale_in_triggered = False
                            scale_in_side = None
                            scale_in_reason = ""

                            if position['status'] == 'long' and support_found:
                                scale_in_triggered = True
                                scale_in_side = 'buy'
                                scale_in_reason = "Large support wall detected below current price."
                            elif position['status'] == 'short' and resistance_found:
                                scale_in_triggered = True                                              scale_in_side = 'sell'
                                scale_in_reason = "Large resistance wall detected above current price."

                            if scale_in_triggered:
                                log_info(f"{NEON_CYAN}OB Scale-In Opportunity Detected! Reason: {scale_in_reason}{RESET}")
                                ob_action_taken_this_cycle = True

                                # Ensure original quantity was stored
                                if position.get('original_quantity') is None:
                                     position['original_quantity'] = position['quantity'] # Store it now if missing                           
                                original_qty = position.get('original_quantity', position['quantity']) # Fallback                                                             if original_qty is None or original_qty <= 0:
                                     log_error("Cannot scale in: Original quantity not available or invalid.")                                                                else:
                                    additional_quantity = original_qty * OB_SCALE_IN_AMOUNT_MULT
                                    log_info(f"Attempting to scale into {position['status']} position by adding {additional_quantity:.{amount_precision_digits}f} (Side: {scale_in_side})")
                                                                                                           # Place additional market order (NOT reduceOnly)
                                    scale_order_result = place_market_order(exchange, symbol, scale_in_side, additional_quantity, reduce_only=False)
                                    is_filled = scale_order_result and (scale_order_result.get('status') == 'closed' or (scale_order_result.get('status') == 'open' and scale_order_result.get('filled', 0) > 0))

                                    if is_filled:
                                        filled_add_qty = scale_order_result.get('filled', additional_quantity)
                                        filled_add_price = scale_order_result.get('average', current_price)
                                        log_info(f"Scale-in order filled/processed: ID {scale_order_result.get('id', 'N/A')} at ~{filled_add_price:.{price_precision_digits}f} for {filled_add_qty:.{amount_precision_digits}f}")

                                        # --- Update Position State ---
                                        old_total_value = position['entry_price'] * position['quantity']
                                        added_value = filled_add_price * filled_add_qty
                                        new_total_quantity = position['quantity'] + filled_add_qty
                                        new_average_entry_price = (old_total_value + added_value) / new_total_quantity if new_total_quantity > 0 else position['entry_price']

                                        position['quantity'] = new_total_quantity
                                        position['entry_price'] = new_average_entry_price
                                        position['times_scaled_in'] = position.get('times_scaled_in', 0) + 1
                                        position['last_ob_action_time'] = pd.Timestamp.now(tz='UTC') # Record action time                                                             # Original quantity remains the same                                                                                                          # Keep existing SL/TP orders and IDs - manual adjustment might be needed                                                                      log_info(f"Position state updated after scale-in: New Qty={new_total_quantity:.{amount_precision_digits}f}, New Avg Entry={new_average_entry_price:.{price_precision_digits}f}")
                                        log_warning("SL/TP orders were NOT automatically adjusted after scale-in. Original SL/TP targets remain.")                                                                                                           save_position_state()
                                        display_position_status(position, price_precision_digits, amount_precision_digits)                    
                                    else:                                                                      log_error(f"Failed to place or confirm fill for scale-in order. Result: {scale_order_result}")
                        # --- End Scale-In Logic ---
                                                                                           except Exception as ob_e:
                         # Errors during fetch/analysis are handled inside the function
                         # This catches unexpected errors in the logic block itself                                                                                    log_error(f"Error during order book action processing: {ob_e}", exc_info=True)
                                                                                       elif is_profitable and not can_take_ob_action:
                     pass # Logged already inside the time check
                elif not is_profitable:
                     log_debug("Position not currently profitable. Skipping order book actions.")
                                                                                                                                                          # C. Check for Indicator-Based Exits (Only if no OB action taken this cycle)                                                                  if not ob_action_taken_this_cycle:
                execute_indicator_exit = False
                exit_reason = ""
                # ... (rest of the existing indicator exit logic remains the same) ...
                if position['status'] == 'long' and last_rsi > rsi_overbought:
                    exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} > {rsi_overbought})"
                    execute_indicator_exit = True
                elif position['status'] == 'short' and last_rsi < rsi_oversold:
                    exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} < {rsi_oversold})"
                    execute_indicator_exit = True                      
                if execute_indicator_exit:
                    # ... (Existing logic to cancel SL/TP and place market exit order) ...                                                                        # IMPORTANT: Ensure this logic also resets the new state fields:
                    # 'original_quantity', 'times_scaled_in', 'last_ob_action_time' to None/0
                    # The existing `default_state = {k: None for k in position.keys()}` reset should handle this.
                    pass # Placeholder for existing logic
                else:
                    # If no indicator exit, just log monitoring status
                    log_info(f"Monitoring {position['status'].upper()} position. Waiting for SL/TP ({position.get('sl_order_id') or 'N/A'}/{position.get('tp_order_id') or 'N/A'}) or TSL update.")
                                                                       
        # --- Check for NEW ENTRIES (only if not currently in a position) ---
        else: # position['status'] is None                                         # ... (Existing entry logic remains the same) ...
            # IMPORTANT: When a new entry is made, initialize the new state fields:
            # Inside the `if is_filled:` block after placing the initial entry order and SL/TP:
            #   position.update({                                                  #       ... (existing fields) ...
            #       'original_quantity': filled_quantity, # Store initial quantity
            #       'times_scaled_in': 0,                                          #       'last_ob_action_time': None
            #   })
            pass # Placeholder for existing logic

        # 7. Wait for the next cycle                                           # ... (Existing sleep logic) ...

# --- Ensure state reset logic handles new fields ---                  # In `check_position_and_orders`, when resetting state:
# The line `default_state = {k: None for k in position.keys()}` should correctly
# capture the new keys and set them to None (or default value if modified).
# Ensure `times_scaled_in` defaults to 0, not None, if using the loop counter approach.
# Let's refine the reset:                                              # Inside `check_position_and_orders`, replace the simple reset with:   #
#             log_info("Resetting local position state.")
#             position['status'] = None                                #             position['entry_price'] = None
#             position['quantity'] = None
#             position['order_id'] = None
#             position['stop_loss'] = None
#             position['take_profit'] = None                           #             position['entry_time'] = None
#             position['sl_order_id'] = None
#             position['tp_order_id'] = None
#             position['highest_price_since_entry'] = None             #             position['lowest_price_since_entry'] = None
#             position['current_trailing_sl_price'] = None
#             # Reset new fields
#             position['original_quantity'] = None                     #             position['times_scaled_in'] = 0
#             position['last_ob_action_time'] = None
#             save_position_state() # Save the reset state
#             display_position_status(position, price_precision_digits, amount_precision_digits) # Show updated status                        #             return True # Indicate position was reset

# Similarly, ensure the fallback indicator exit logic uses this detailed reset.
# And ensure the OB early exit logic uses this detailed reset.
python
# --- Upgrade Snippet 3: Limit Order Entry ---
# Add this near configuration loading
ENTRY_ORDER_TYPE: str = config.get("entry_order_type", "market").lower() # "market" or "limit"
LIMIT_ORDER_PRICE_OFFSET_ATR_MULT: float = float(config.get("limit_order_price_offset_atr_mult", 0.1)) # e.g., 0.1 * ATR away from current price
LIMIT_ORDER_EXPIRY_SECONDS: int = int(config.get("limit_order_expiry_seconds", 120)) # How long to wait for fill

if ENTRY_ORDER_TYPE == "limit":
    log_info(f"Using LIMIT orders for entry (Offset: {LIMIT_ORDER_PRICE_OFFSET_ATR_MULT}x ATR, Expiry: {LIMIT_ORDER_EXPIRY_SECONDS}s)")
    if not needs_atr: # Ensure ATR is calculated if using limit orders based on it
        log_warning("ATR calculation is suggested when using ATR-based limit order offsets. Ensure 'enable_atr_sl_tp' or 'enable_trailing_stop' is true or modify logic.")
elif ENTRY_ORDER_TYPE != "market":
    log_error(f"Invalid 'entry_order_type' in config: '{ENTRY_ORDER_TYPE}'. Use 'market' or 'limit'. Defaulting to market.")
    ENTRY_ORDER_TYPE = "market"

# --- New Order Placement Function for Limit Orders (Decorated) ---
@api_retry_decorator
def place_limit_order(exchange_instance: ccxt.Exchange, trading_symbol: str, side: str, amount: float, price: float, reduce_only: bool = False) -> Optional[Dict[str, Any]]:
    """Places a limit order ('buy' or 'sell') with retry logic. Optionally reduceOnly."""
    if side not in ['buy', 'sell']:
        log_error(f"Invalid order side: '{side}'.")
        return None
    if amount <= 0 or price <= 0:
        log_error(f"Invalid order amount ({amount}) or price ({price}).")
        return None

    try:
        market = exchange_instance.market(trading_symbol)
        base_currency: str = market.get('base', '')
        amount_formatted = float(exchange_instance.amount_to_precision(trading_symbol, amount))
        price_formatted = float(exchange_instance.price_to_precision(trading_symbol, price))

        log_info(f"Attempting to place {side.upper()} LIMIT order for {amount_formatted:.{amount_precision_digits}f} {base_currency} @ {price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if reduce_only else ''}...")

        params = {'type': 'limit'} # Ensure type is specified
        if reduce_only and (market.get('swap') or market.get('future') or market.get('contract')):
            params['reduceOnly'] = True
            log_debug("Applying 'reduceOnly=True' to limit order.")

        if SIMULATION_MODE:
            log_warning("!!! SIMULATION: Limit order placement skipped.")
            order_id = f'sim_limit_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
            # Simulate an open order initially
            order = {
                'id': order_id, 'clientOrderId': order_id, 'timestamp': int(time.time() * 1000),
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'open', # Limit orders start as open
                'symbol': trading_symbol, 'type': 'limit', 'timeInForce': 'GTC', # Good 'Til Canceled is common default
                'side': side, 'price': price_formatted, 'average': None, # Not filled yet
                'amount': amount_formatted, 'filled': 0.0, 'remaining': amount_formatted,
                'cost': 0.0, 'fee': None, 'info': {'simulated': True, 'reduceOnly': params.get('reduceOnly', False)}
            }
        else:
            # --- LIVE TRADING ---
            log_warning(f"!!! LIVE MODE: Placing real limit order {'(ReduceOnly)' if params.get('reduceOnly') else ''}.")
            # Use create_limit_order or create_order with type='limit'
            order = exchange_instance.create_limit_order(trading_symbol, side, amount_formatted, price_formatted, params=params)
            # --- END LIVE TRADING ---

        log_info(f"{side.capitalize()} limit order request processed {'(Simulated)' if SIMULATION_MODE else ''}.")
        order_id: Optional[str] = order.get('id')
        order_status: Optional[str] = order.get('status')
        log_info(f"Order Result | ID: {order_id or 'N/A'}, Status: {order_status or 'N/A'}, Price: {price_formatted:.{price_precision_digits}f}")
        # Add short delay
        time.sleep(0.5) # Shorter delay for limit order placement is usually fine
        return order

    except ccxt.InsufficientFunds as e:
        log_error(f"Insufficient funds for limit {side} {amount} {trading_symbol} @ {price}. Error: {e}")
        return None
    except ccxt.InvalidOrder as e:
         log_error(f"Invalid limit order parameters for {side} {amount} {trading_symbol} @ {price}: {e}")
         return None
    except ccxt.ExchangeError as e:
        log_error(f"Exchange specific error placing {side} limit order for {trading_symbol}: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error placing {side} limit order: {e}", exc_info=True)
        return None

# --- Modify the main trading logic where entry orders are placed ---
# Inside the signal checking block (e.g., if long_signal and position['status'] is None):
#   ... calculate stop_loss_price, take_profit_price, quantity ...
#   if quantity:
#       if ENTRY_ORDER_TYPE == "limit":
#           # Calculate limit price (e.g., slightly better than current price based on ATR)
#           atr_value = df[atr_col_name_expected].iloc[-1] if atr_col_name_expected in df.columns else 0
#           price_offset = atr_value * LIMIT_ORDER_PRICE_OFFSET_ATR_MULT if atr_value > 0 else current_price * 0.0005 # Fallback small offset
#
#           limit_buy_price = current_price - price_offset if side == 'buy' else current_price + price_offset
#           limit_buy_price = float(exchange.price_to_precision(symbol, limit_buy_price)) # Adjust for precision
#
#           log_info(f"Attempting LIMIT BUY entry at {limit_buy_price:.{price_precision_digits}f}")
#           entry_order = place_limit_order(exchange, symbol, 'buy', quantity, limit_buy_price)
#
#           if entry_order and entry_order.get('id'):
#               # --- Wait for fill or expiry ---
#               entry_order_id = entry_order['id']
#               start_wait_time = time.time()
#               filled = False
#               while time.time() - start_wait_time < LIMIT_ORDER_EXPIRY_SECONDS:
#                   try:
#                       @api_retry_decorator # Decorate the check function
#                       def check_order_status(exch, order_id, sym):
#                           return exch.fetch_order(order_id, sym)
#
#                       order_status = check_order_status(exchange, entry_order_id, symbol)
#
#                       if order_status.get('status') == 'closed':
#                           log_info(f"Limit entry order {entry_order_id} filled!")
#                           position['status'] = 'long'
#                           position['entry_price'] = order_status.get('average', limit_buy_price) # Use avg price if available
#                           position['quantity'] = order_status.get('filled', quantity)
#                           position['order_id'] = entry_order_id
#                           position['stop_loss'] = stop_loss_price # Set initial SL/TP based on original calculation
#                           position['take_profit'] = take_profit_price
#                           position['entry_time'] = pd.Timestamp.now(tz='UTC')
#                           # Reset TSL fields
#                           position['highest_price_since_entry'] = position['entry_price']
#                           position['lowest_price_since_entry'] = position['entry_price']
#                           position['current_trailing_sl_price'] = None
#                           save_position_state()
#                           # Optional: Place SL/TP orders now if strategy dictates
#                           filled = True
#                           break # Exit wait loop
#                       elif order_status.get('status') in ['canceled', 'rejected', 'expired']:
#                           log_warning(f"Limit entry order {entry_order_id} was {order_status.get('status')}. No entry.")
#                           break # Exit wait loop
#                       else: # Still open or unknown
#                           log_debug(f"Limit order {entry_order_id} status: {order_status.get('status', 'unknown')}. Waiting...")
#                           time.sleep(5) # Check every 5 seconds
#
#                   except ccxt.OrderNotFound:
#                        log_error(f"OrderNotFound when checking limit order {entry_order_id}. Assuming canceled/failed.")
#                        break
#                   except Exception as e:
#                        log_error(f"Error checking limit order status: {e}", exc_info=True)
#                        # Decide whether to break or continue waiting after an error
#                        time.sleep(10) # Wait longer after error
#
#               if not filled:
#                   log_warning(f"Limit entry order {entry_order_id} did not fill within {LIMIT_ORDER_EXPIRY_SECONDS}s. Cancelling.")
#                   try:
#                       cancel_order_with_retry(exchange, entry_order_id, symbol)
#                   except Exception as cancel_e:
#                       log_error(f"Failed to cancel unfilled limit order {entry_order_id}: {cancel_e}")
#           else:
#               log_error("Failed to place limit entry order.")
#
#       elif ENTRY_ORDER_TYPE == "market":
#           # --- Original Market Order Logic ---
#           log_info(f"Attempting MARKET BUY entry...")
#           entry_order = place_market_order(exchange, symbol, 'buy', quantity)
#           if entry_order and entry_order.get('status') == 'closed': # Market orders usually fill instantly
#               # ... (update position state as in the original script) ...
#               save_position_state()
#           else:
#               log_error("Market entry order failed or did not fill immediately.")
#       # ... (rest of the logic for short side) ...

```

```python
# --- Upgrade Snippet 4: Configuration Hot-Reloading ---
# Add near constants/config
CONFIG_LAST_MODIFIED_TIME: float = 0.0
CHECK_CONFIG_INTERVAL_SECONDS: int = 60 # How often to check the config file for changes

# --- Function to check and reload config ---
def check_and_reload_config(filename: str = CONFIG_FILE) -> bool:
    """Checks if the config file has been modified and reloads it if necessary."""
    global config, CONFIG_LAST_MODIFIED_TIME
    global api_key, secret, passphrase # Need to potentially reload these too if changed
    global exchange_id, symbol, timeframe, rsi_length, rsi_overbought, rsi_oversold # etc.
    global risk_percentage, stop_loss_percentage, take_profit_percentage # etc.
    global enable_atr_sl_tp, atr_length, atr_sl_multiplier, atr_tp_multiplier # etc.
    global enable_trailing_stop, trailing_stop_atr_multiplier, trailing_stop_activation_atr_multiplier # etc.
    global ob_volume_threshold_multiplier, ob_lookback # etc.
    global entry_volume_confirmation_enabled, entry_volume_ma_length, entry_volume_multiplier # etc.
    global sleep_interval_seconds, SIMULATION_MODE # etc.
    # Add ALL parameters loaded from config that might change

    try:
        current_mod_time = os.path.getmtime(filename)
        if current_mod_time > CONFIG_LAST_MODIFIED_TIME:
            log_warning(f"Detected change in configuration file '{filename}'. Reloading...")
            try:
                new_config = load_config(filename) # Use the existing load function

                # --- Update global variables ---
                # Critical ones first (exchange/symbol change might require restart, handle carefully)
                new_exchange_id = new_config.get("exchange_id", "bybit").lower()
                new_symbol = new_config.get("symbol", "").strip().upper()

                if new_exchange_id != exchange_id or new_symbol != symbol:
                    # Simplest approach: Log and exit, requiring manual restart
                    log_error(f"CRITICAL CHANGE DETECTED: Exchange ID or Symbol changed in config ('{exchange_id}'->'{new_exchange_id}', '{symbol}'->'{new_symbol}'). Restart required.")
                    # Optionally: Implement logic to close existing position and re-initialize exchange connection
                    # This is complex and risky, exiting might be safer.
                    send_telegram_message(f"⚠️ CONFIG CHANGE REQUIRES RESTART: Exchange/Symbol changed.") # If using Telegram
                    sys.exit(1) # Force exit

                # Update other parameters (Example - add all relevant ones)
                config = new_config # Store the new config dict
                timeframe = config.get("timeframe", "1h")
                rsi_length = int(config.get("rsi_length", 14))
                risk_percentage = float(config.get("risk_percentage", 0.01))
                enable_trailing_stop = config.get("enable_trailing_stop", False)
                trailing_stop_atr_multiplier = float(config.get("trailing_stop_atr_multiplier", 1.5))
                sleep_interval_seconds = int(config.get("sleep_interval_seconds", 900))
                new_sim_mode = config.get("simulation_mode", True)

                # Log changes for important parameters
                if new_sim_mode != SIMULATION_MODE:
                     log_warning(f"SIMULATION MODE changed to: {'ACTIVE' if new_sim_mode else '!!! LIVE TRADING !!!'}")
                     send_telegram_message(f"⚠️ SIMULATION MODE changed to: {'ON' if new_sim_mode else 'OFF (LIVE!)'}") # If using Telegram
                SIMULATION_MODE = new_sim_mode

                # Add logging for other parameter changes if desired
                log_info(f"Risk % updated to: {risk_percentage*100:.2f}%")
                log_info(f"Trailing Stop enabled: {enable_trailing_stop}, ATR Multiplier: {trailing_stop_atr_multiplier}")
                log_info(f"Sleep Interval updated to: {sleep_interval_seconds}s")
                # ... log other reloaded params ...

                CONFIG_LAST_MODIFIED_TIME = current_mod_time
                log_info("Configuration reloaded successfully.")
                return True # Indicate reload happened

            except Exception as e:
                log_error(f"Failed to reload configuration from '{filename}': {e}", exc_info=True)
                # Keep using the old config
                return False # Indicate reload failed
        else:
            # log_debug("Config file unchanged.") # Optional: Log check
            return False # Indicate no reload needed

    except FileNotFoundError:
        log_error(f"Configuration file '{filename}' not found during check. Cannot reload.")
        # Keep using the current config in memory
        return False
    except Exception as e:
        log_error(f"Error checking config file modification time: {e}", exc_info=True)
        return False


# --- Modify the main trading loop ---
# Inside the `while True:` loop, before fetching data or sleeping:

    # --- Check for Config Reload ---
    current_time = time.time()
    # Use a separate timer/variable to track when to check the config file
    # Initialize `last_config_check_time = 0` before the main loop
    global last_config_check_time # Assuming defined outside the loop
    if current_time - last_config_check_time > CHECK_CONFIG_INTERVAL_SECONDS:
        check_and_reload_config()
        last_config_check_time = current_time # Update last check time regardless of outcome

    # --- Fetch Data ---
    log_debug("Fetching latest market data...")
    # ... rest of the loop (fetch_ohlcv_data, calculate_indicators, etc.) ...

    # --- Sleep ---
    print_cycle_divider(pd.Timestamp.now(tz='UTC'))
    log_info(f"Cycle complete. Sleeping for {sleep_interval_seconds} seconds...")
    neon_sleep_timer(sleep_interval_seconds) # Use the existing timer

# --- Initialization before the loop ---
# Make sure to initialize CONFIG_LAST_MODIFIED_TIME after the initial load
try:
    CONFIG_LAST_MODIFIED_TIME = os.path.getmtime(CONFIG_FILE)
except FileNotFoundError:
    CONFIG_LAST_MODIFIED_TIME = 0 # Should not happen if initial load worked

last_config_check_time = 0 # Initialize check timer

# --- Start main loop ---
# while True:
#    ... (config check added inside) ...


python
# trading_bot_neon_enhanced_v2.py
# Enhanced version incorporating ATR SL/TP, Trailing Stops, Config File,
# Volume Confirmation, and Retry Logic.

import ccxt
import os
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import pandas_ta as ta # Using pandas_ta for indicator calculations
import json # For config, pretty printing order details and saving state
import os.path # For checking if state file exists
from typing import Optional, Tuple, Dict, Any, List, Union
import functools # For retry decorator
import sys # For exit

# --- Colorama Initialization and Neon Palette ---
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True) # Autoreset ensures styles don't leak

    # Define neon color palette
    NEON_GREEN = Fore.GREEN + Style.BRIGHT
    NEON_PINK = Fore.MAGENTA + Style.BRIGHT
    NEON_CYAN = Fore.CYAN + Style.BRIGHT
    NEON_RED = Fore.RED + Style.BRIGHT
    NEON_YELLOW = Fore.YELLOW + Style.BRIGHT
    NEON_BLUE = Fore.BLUE + Style.BRIGHT
    RESET = Style.RESET_ALL # Although autoreset is on, good practice
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Warning: colorama not found. Neon styling will be disabled. Install with: pip install colorama")
    # Define dummy colors if colorama is not available
    NEON_GREEN = NEON_PINK = NEON_CYAN = NEON_RED = NEON_YELLOW = NEON_BLUE = RESET = ""
    COLORAMA_AVAILABLE = False

# --- Logging Configuration ---
log_format_base: str = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
# Configure root logger - handlers can be added later if needed (e.g., file handler)
logging.basicConfig(level=logging.INFO, format=log_format_base, datefmt='%Y-%m-%d %H:%M:%S')
logger: logging.Logger = logging.getLogger(__name__) # Get logger for this module

# --- Neon Display Functions ---

def print_neon_header():
    """Prints a neon-styled header banner."""
    print(f"{NEON_CYAN}{'=' * 70}{RESET}")
    print(f"{NEON_PINK}{Style.BRIGHT}     Enhanced RSI/OB Trader Neon Bot - Configurable v2     {RESET}")
    print(f"{NEON_CYAN}{'=' * 70}{RESET}")

def display_error_box(message: str):
    """Displays an error message in a neon box."""
    box_width = 70
    print(f"{NEON_RED}{'!' * box_width}{RESET}")
    print(f"{NEON_RED}! {message:^{box_width-4}} !{RESET}")
    print(f"{NEON_RED}{'!' * box_width}{RESET}")

def display_warning_box(message: str):
    """Displays a warning message in a neon box."""
    box_width = 70
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")
    print(f"{NEON_YELLOW}~ {message:^{box_width-4}} ~{RESET}")
    print(f"{NEON_YELLOW}{'~' * box_width}{RESET}")

# Custom logging wrappers with colorama
def log_info(msg: str):
    logger.info(f"{NEON_GREEN}{msg}{RESET}")

def log_error(msg: str, exc_info=False):
    # Show first line in box for prominence, log full message
    first_line = msg.split('\n', 1)[0]
    display_error_box(first_line)
    logger.error(f"{NEON_RED}{msg}{RESET}", exc_info=exc_info)

def log_warning(msg: str):
    display_warning_box(msg)
    logger.warning(f"{NEON_YELLOW}{msg}{RESET}")

def log_debug(msg: str):
     # Use a less prominent color for debug
    logger.debug(f"{Fore.WHITE}{msg}{RESET}") # Simple white for debug

def print_cycle_divider(timestamp: pd.Timestamp):
    """Prints a neon divider for each trading cycle."""
    box_width = 70
    print(f"\n{NEON_BLUE}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}Cycle Start: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{NEON_BLUE}{'=' * box_width}{RESET}")

def display_position_status(position: Dict[str, Any], price_precision: int = 4, amount_precision: int = 8):
    """Displays position status with neon colors and formatted values."""
    status = position.get('status', None)
    entry_price = position.get('entry_price')
    quantity = position.get('quantity')
    sl = position.get('stop_loss')
    tp = position.get('take_profit')
    tsl = position.get('current_trailing_sl_price') # Added TSL display

    entry_str = f"{entry_price:.{price_precision}f}" if isinstance(entry_price, (float, int)) else "N/A"
    qty_str = f"{quantity:.{amount_precision}f}" if isinstance(quantity, (float, int)) else "N/A"
    sl_str = f"{sl:.{price_precision}f}" if isinstance(sl, (float, int)) else "N/A"
    tp_str = f"{tp:.{price_precision}f}" if isinstance(tp, (float, int)) else "N/A"
    tsl_str = f" | TSL: {tsl:.{price_precision}f}" if isinstance(tsl, (float, int)) else "" # Show TSL only if active

    if status == 'long':
        color = NEON_GREEN
        status_text = "LONG"
    elif status == 'short':
        color = NEON_RED
        status_text = "SHORT"
    else:
        color = NEON_CYAN
        status_text = "None"

    print(f"{color}Position Status: {status_text}{RESET} | Entry: {entry_str} | Qty: {qty_str} | SL: {sl_str} | TP: {tp_str}{tsl_str}")


def display_market_stats(current_price: float, rsi: float, stoch_k: float, stoch_d: float, atr: Optional[float], price_precision: int):
    """Displays market stats in a neon-styled panel."""
    print(f"{NEON_PINK}--- Market Stats ---{RESET}")
    print(f"{NEON_GREEN}Price:{RESET}  {current_price:.{price_precision}f}")
    print(f"{NEON_CYAN}RSI:{RESET}    {rsi:.2f}")
    print(f"{NEON_YELLOW}StochK:{RESET} {stoch_k:.2f}")
    print(f"{NEON_YELLOW}StochD:{RESET} {stoch_d:.2f}")
    # Use price_precision for ATR display as it's a price volatility measure
    if atr is not None:
         print(f"{NEON_BLUE}ATR:{RESET}    {atr:.{price_precision}f}") # Display ATR
    print(f"{NEON_PINK}--------------------{RESET}")

def display_order_blocks(bullish_ob: Optional[Dict], bearish_ob: Optional[Dict], price_precision: int):
    """Displays order blocks with neon colors."""
    found = False
    if bullish_ob:
        print(f"{NEON_GREEN}Bullish OB:{RESET} {bullish_ob['time'].strftime('%H:%M')} | Low: {bullish_ob['low']:.{price_precision}f} | High: {bullish_ob['high']:.{price_precision}f}")
        found = True
    if bearish_ob:
        print(f"{NEON_RED}Bearish OB:{RESET} {bearish_ob['time'].strftime('%H:%M')} | Low: {bearish_ob['low']:.{price_precision}f} | High: {bearish_ob['high']:.{price_precision}f}")
        found = True
    if not found:
        print(f"{NEON_BLUE}Order Blocks: None detected in recent data.{RESET}")


def display_signal(signal_type: str, direction: str, reason: str):
    """Displays trading signals with neon colors."""
    if direction.lower() == 'long':
        color = NEON_GREEN
    elif direction.lower() == 'short':
        color = NEON_RED
    else:
        color = NEON_YELLOW # For general signals/alerts

    print(f"{color}{Style.BRIGHT}*** {signal_type.upper()} {direction.upper()} SIGNAL ***{RESET}\n   Reason: {reason}")

def neon_sleep_timer(seconds: int):
    """Displays a neon countdown timer."""
    if not COLORAMA_AVAILABLE or seconds <= 0: # Fallback if colorama not installed or zero sleep
        if seconds > 0:
            print(f"Sleeping for {seconds} seconds...")
            time.sleep(seconds)
        return

    interval = 0.5 # Update interval for the timer display
    steps = int(seconds / interval)
    for i in range(steps, -1, -1):
        remaining_seconds = max(0, int(i * interval)) # Ensure non-negative
        # Flashing effect for last 5 seconds
        color = NEON_RED if remaining_seconds <= 5 and i % 2 == 0 else NEON_YELLOW
        print(f"{color}Next cycle in: {remaining_seconds} seconds... {Style.RESET_ALL}", end='\r')
        time.sleep(interval)
    print(" " * 50, end='\r')  # Clear line after countdown

def print_shutdown_message():
    """Prints a neon shutdown message."""
    box_width = 70
    print(f"\n{NEON_PINK}{'=' * box_width}{RESET}")
    print(f"{NEON_CYAN}{Style.BRIGHT}{'RSI/OB Trader Bot Stopped - Goodbye!':^{box_width}}{RESET}")
    print(f"{NEON_PINK}{'=' * box_width}{RESET}")


# --- Constants ---
DEFAULT_PRICE_PRECISION: int = 4
DEFAULT_AMOUNT_PRECISION: int = 8
POSITION_STATE_FILE = 'position_state.json' # Define filename for state persistence
CONFIG_FILE = 'config.json'

# --- Retry Decorator ---
RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout
)

def retry_api_call(max_retries: int = 3, initial_delay: float = 5.0, backoff_factor: float = 2.0):
    """Decorator factory to create retry decorators with specific parameters."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    retries += 1
                    if retries >= max_retries:
                        log_error(f"API call '{func.__name__}' failed after {max_retries} retries. Last error: {e}", exc_info=False)
                        raise # Re-raise the last exception
                    else:
                        log_warning(f"API call '{func.__name__}' failed with {type(e).__name__}. Retrying in {delay:.1f}s... (Attempt {retries}/{max_retries})")
                        # Use neon sleep if available, otherwise time.sleep
                        try:
                            neon_sleep_timer(int(delay))
                        except NameError:
                            time.sleep(delay)
                        delay *= backoff_factor # Exponential backoff
                except Exception as e:
                     # Handle non-retryable exceptions immediately
                     log_error(f"Non-retryable error in API call '{func.__name__}': {e}", exc_info=True)
                     raise # Re-raise immediately
            # This line should theoretically not be reached if max_retries > 0
            # but added for safety to ensure function returns or raises.
            raise RuntimeError(f"API call '{func.__name__}' failed unexpectedly after retries.")
        return wrapper
    return decorator


# --- Configuration Loading ---
def load_config(filename: str = CONFIG_FILE) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    log_info(f"Attempting to load configuration from '{filename}'...")
    try:
        with open(filename, 'r') as f:
            config_data = json.load(f)
        log_info(f"Configuration loaded successfully from {filename}")
        # Basic validation (presence of essential keys)
        required_keys = ["exchange_id", "symbol", "timeframe", "risk_percentage", "simulation_mode"]
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
             log_error(f"CRITICAL: Missing required configuration keys in '{filename}': {missing_keys}")
             sys.exit(1) # Use sys.exit for clean exit
        # TODO: Add more specific validation (types, ranges) if needed
        return config_data
    except FileNotFoundError:
        log_error(f"CRITICAL: Configuration file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log_error(f"CRITICAL: Error decoding JSON from configuration file '{filename}': {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"CRITICAL: An unexpected error occurred loading configuration: {e}", exc_info=True)
        sys.exit(1)

# Load config early
config = load_config()

# Create the decorator instance using config values
api_retry_decorator = retry_api_call(
    max_retries=config.get("retry_max_retries", 3),
    initial_delay=config.get("retry_initial_delay", 5.0),
    backoff_factor=config.get("retry_backoff_factor", 2.0)
)

# --- Environment & Exchange Setup ---
print_neon_header() # Show header early
load_dotenv()
log_info("Attempting to load environment variables from .env file (for API keys)...")

exchange_id: str = config.get("exchange_id", "bybit").lower()
api_key_env_var: str = f"{exchange_id.upper()}_API_KEY"
secret_key_env_var: str = f"{exchange_id.upper()}_SECRET_KEY"
passphrase_env_var: str = f"{exchange_id.upper()}_PASSPHRASE" # Some exchanges like kucoin, okx use this

api_key: Optional[str] = os.getenv(api_key_env_var)
secret: Optional[str] = os.getenv(secret_key_env_var)
passphrase: str = os.getenv(passphrase_env_var, '')

if not api_key or not secret:
    log_error(f"CRITICAL: API Key or Secret not found using env vars '{api_key_env_var}' and '{secret_key_env_var}'. "
              f"Please ensure these exist in your .env file or environment.")
    sys.exit(1)

log_info(f"Attempting to connect to exchange: {exchange_id}")
exchange: ccxt.Exchange
try:
    exchange_class = getattr(ccxt, exchange_id)
    exchange_config: Dict[str, Any] = {
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True, # CCXT internal rate limiting
        'options': {
            # Adjust 'swap' or 'spot' based on your market type in config/symbol
            'defaultType': 'swap' if ':' in config.get("symbol", "") else 'spot',
            # 'adjustForTimeDifference': True, # Optional: Helps with timestamp issues
        }
    }
    if passphrase:
        log_info("Passphrase detected, adding to exchange configuration.")
        exchange_config['password'] = passphrase

    exchange = exchange_class(exchange_config)

    # Load markets with retry mechanism applied
    @api_retry_decorator
    def load_markets_with_retry(exch_instance):
         log_info("Loading markets...")
         exch_instance.load_markets()

    load_markets_with_retry(exchange)

    log_info(f"Successfully connected to {exchange_id}. Markets loaded ({len(exchange.markets)} symbols found).")

except ccxt.AuthenticationError as e:
    log_error(f"Authentication failed connecting to {exchange_id}. Check API Key/Secret/Passphrase. Error: {e}")
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    log_error(f"Exchange {exchange_id} is not available. Error: {e}")
    sys.exit(1)
except AttributeError:
    log_error(f"Exchange ID '{exchange_id}' not found in ccxt library.")
    sys.exit(1)
except Exception as e:
    # Check if it's a retryable error that failed all retries (already logged by decorator)
    if not isinstance(e, RETRYABLE_EXCEPTIONS):
        log_error(f"An unexpected error occurred during exchange initialization or market loading: {e}", exc_info=True)
    sys.exit(1)


# --- Trading Parameters (Loaded from config) ---
symbol: str = config.get("symbol", "").strip().upper()
if not symbol:
    log_error("CRITICAL: 'symbol' not specified in config.json")
    sys.exit(1)

# Validate symbol against exchange markets and get precision info
try:
    if symbol not in exchange.markets:
        log_warning(f"Symbol '{symbol}' from config not found or not supported on {exchange_id}.")
        available_symbols: List[str] = list(exchange.markets.keys())
        # Filter for potential swap markets if needed
        if exchange_config.get('options', {}).get('defaultType') == 'swap':
             available_symbols = [s for s in available_symbols if ':' in s or exchange.markets[s].get('swap')]
        log_info(f"Some available symbols ({len(available_symbols)} total): {available_symbols[:15]}...")
        sys.exit(1)

    log_info(f"Using trading symbol from config: {symbol}")
    # Store precision info globally for convenience
    market_info = exchange.markets[symbol]
    price_precision_digits: int = int(market_info.get('precision', {}).get('price', DEFAULT_PRICE_PRECISION))
    amount_precision_digits: int = int(market_info.get('precision', {}).get('amount', DEFAULT_AMOUNT_PRECISION))
    min_tick: float = 1 / (10 ** price_precision_digits) if price_precision_digits > 0 else 0.01 # Smallest price change

    log_info(f"Symbol Precision | Price: {price_precision_digits} decimals (Min Tick: {min_tick}), Amount: {amount_precision_digits} decimals")

except Exception as e:
    log_error(f"An error occurred while validating the symbol or getting precision: {e}", exc_info=True)
    sys.exit(1)

timeframe: str = config.get("timeframe", "1h")
rsi_length: int = int(config.get("rsi_length", 14))
rsi_overbought: int = int(config.get("rsi_overbought", 70))
rsi_oversold: int = int(config.get("rsi_oversold", 30))
stoch_k: int = int(config.get("stoch_k", 14))
stoch_d: int = int(config.get("stoch_d", 3))
stoch_smooth_k: int = int(config.get("stoch_smooth_k", 3))
stoch_overbought: int = int(config.get("stoch_overbought", 80))
stoch_oversold: int = int(config.get("stoch_oversold", 20))
data_limit: int = int(config.get("data_limit", 200))
sleep_interval_seconds: int = int(config.get("sleep_interval_seconds", 900))
risk_percentage: float = float(config.get("risk_percentage", 0.01))

# SL/TP and Trailing Stop Parameters (Conditional loading)
enable_atr_sl_tp: bool = config.get("enable_atr_sl_tp", False)
enable_trailing_stop: bool = config.get("enable_trailing_stop", False)
atr_length: int = int(config.get("atr_length", 14)) # Needed if either ATR SL/TP or TSL is enabled

# Validate ATR length if needed
needs_atr = enable_atr_sl_tp or enable_trailing_stop
if needs_atr and atr_length <= 0:
    log_error(f"CRITICAL: 'atr_length' must be positive ({atr_length}) if ATR SL/TP or Trailing Stop is enabled.")
    sys.exit(1)

if enable_atr_sl_tp:
    atr_sl_multiplier: float = float(config.get("atr_sl_multiplier", 2.0))
    atr_tp_multiplier: float = float(config.get("atr_tp_multiplier", 3.0))
    log_info(f"Using ATR-based Stop Loss ({atr_sl_multiplier}x ATR) and Take Profit ({atr_tp_multiplier}x ATR).")
    if atr_sl_multiplier <= 0 or atr_tp_multiplier <= 0:
         log_error("CRITICAL: ATR SL/TP multipliers must be positive.")
         sys.exit(1)
else:
    stop_loss_percentage: float = float(config.get("stop_loss_percentage", 0.02))
    take_profit_percentage: float = float(config.get("take_profit_percentage", 0.04))
    log_info(f"Using Fixed Percentage Stop Loss ({stop_loss_percentage*100:.1f}%) and Take Profit ({take_profit_percentage*100:.1f}%).")
    if stop_loss_percentage <= 0 or take_profit_percentage <= 0:
         log_error("CRITICAL: Fixed SL/TP percentages must be positive.")
         sys.exit(1)

if enable_trailing_stop:
    trailing_stop_atr_multiplier: float = float(config.get("trailing_stop_atr_multiplier", 1.5))
    trailing_stop_activation_atr_multiplier: float = float(config.get("trailing_stop_activation_atr_multiplier", 1.0))
    log_info(f"Trailing Stop Loss is ENABLED (Activate @ {trailing_stop_activation_atr_multiplier}x ATR profit, Trail @ {trailing_stop_atr_multiplier}x ATR).")
    if trailing_stop_atr_multiplier <= 0 or trailing_stop_activation_atr_multiplier < 0: # Activation can be 0
         log_error("CRITICAL: Trailing stop ATR multiplier must be positive, activation multiplier non-negative.")
         sys.exit(1)
else:
    log_info("Trailing Stop Loss is DISABLED.")

# OB Parameters
ob_volume_threshold_multiplier: float = float(config.get("ob_volume_threshold_multiplier", 1.5))
ob_lookback: int = int(config.get("ob_lookback", 10))
if ob_lookback <= 0:
    log_error("CRITICAL: 'ob_lookback' must be positive.")
    sys.exit(1)

# Volume Confirmation Parameters
entry_volume_confirmation_enabled: bool = config.get("entry_volume_confirmation_enabled", True)
entry_volume_ma_length: int = int(config.get("entry_volume_ma_length", 20))
entry_volume_multiplier: float = float(config.get("entry_volume_multiplier", 1.2))
if entry_volume_confirmation_enabled:
    log_info(f"Entry Volume Confirmation: ENABLED (Vol > {entry_volume_multiplier}x MA({entry_volume_ma_length}))")
    if entry_volume_ma_length <= 0 or entry_volume_multiplier <= 0:
         log_error("CRITICAL: Volume MA length and multiplier must be positive.")
         sys.exit(1)
else:
    log_info("Entry Volume Confirmation: DISABLED")

# --- Simulation Mode ---
SIMULATION_MODE = config.get("simulation_mode", True) # Default to TRUE for safety
if SIMULATION_MODE:
    log_warning("SIMULATION MODE IS ACTIVE (set in config.json). No real orders will be placed.")
else:
    log_warning("!!! LIVE TRADING MODE IS ACTIVE (set in config.json). REAL ORDERS WILL BE PLACED. !!!")
    # Add confirmation step for live trading
    try:
        user_confirm = input(f"{NEON_RED}TYPE 'LIVE' TO CONFIRM LIVE TRADING or press Enter to exit: {RESET}")
        if user_confirm.strip().upper() != "LIVE":
            log_info("Live trading not confirmed. Exiting.")
            sys.exit(0)
        log_info("Live trading confirmed by user.")
    except EOFError: # Handle non-interactive environments
         log_error("Cannot get user confirmation in non-interactive mode. Exiting live mode.")
         sys.exit(1)


# --- Position Management State (Includes Trailing Stop fields) ---
position: Dict[str, Any] = {
    'status': None,         # None, 'long', or 'short'
    'entry_price': None,    # Float
    'quantity': None,       # Float
    'order_id': None,       # String (ID of the entry order)
    'stop_loss': None,      # Float (Initial or last manually set SL price)
    'take_profit': None,    # Float (Price level)
    'entry_time': None,     # pd.Timestamp
    'sl_order_id': None,    # String (ID of the open SL order)
    'tp_order_id': None,    # String (ID of the open TP order)
    # Fields for Trailing Stop Loss
    'highest_price_since_entry': None, # For long positions
    'lowest_price_since_entry': None,  # For short positions
    'current_trailing_sl_price': None # Active trailing SL price (can differ from 'stop_loss' if TSL active)
}

# --- State Saving and Resumption Functions ---
def save_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Saves position state to a JSON file."""
    global position
    try:
        # Create a copy to serialize Timestamp safely
        state_to_save = position.copy()
        if isinstance(state_to_save.get('entry_time'), pd.Timestamp):
            # Ensure timezone information is handled correctly (ISO format preserves it)
            state_to_save['entry_time'] = state_to_save['entry_time'].isoformat()

        with open(filename, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        log_debug(f"Position state saved to {filename}") # Use debug for less noise
    except Exception as e:
        log_error(f"Error saving position state to {filename}: {e}", exc_info=True)

def load_position_state(filename: str = POSITION_STATE_FILE) -> None:
    """Loads position state from a JSON file, handling potential missing keys."""
    global position
    log_info(f"Attempting to load position state from '{filename}'...")
    # Use the current global `position` dict as the template for keys and defaults
    default_position_state = position.copy()

    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                loaded_state: Dict[str, Any] = json.load(f)

            # Convert entry_time back to Timestamp if present and not None
            entry_time_str = loaded_state.get('entry_time')
            if entry_time_str:
                try:
                    # Attempt parsing, assume ISO format which includes timezone if saved correctly
                    ts = pd.Timestamp(entry_time_str)
                    # Ensure it's timezone-aware (UTC is preferred)
                    if ts.tzinfo is None:
                       ts = ts.tz_localize('UTC')
                    else:
                       ts = ts.tz_convert('UTC')
                    loaded_state['entry_time'] = ts
                except ValueError:
                    log_error(f"Could not parse entry_time '{entry_time_str}' from state file. Setting to None.")
                    loaded_state['entry_time'] = None

            # Update only the keys that exist in the current `position` structure
            updated_count = 0
            missing_in_file = []
            extra_in_file = []

            current_keys = set(default_position_state.keys())
            loaded_keys = set(loaded_state.keys())

            for key in current_keys:
                 if key in loaded_keys:
                     # Basic type check (optional but recommended)
                     # Ensure loaded value is not None before comparing types if default is None
                     if default_position_state[key] is None or loaded_state[key] is None or isinstance(loaded_state[key], type(default_position_state[key])):
                         position[key] = loaded_state[key]
                         updated_count += 1
                     else:
                        log_warning(f"Type mismatch for key '{key}' in state file. Expected {type(default_position_state[key])}, got {type(loaded_state[key])}. Using default.")
                        position[key] = default_position_state[key] # Reset to default on type mismatch
                 else:
                     missing_in_file.append(key)
                     position[key] = default_position_state[key] # Reset to default if missing in file

            extra_in_file = list(loaded_keys - current_keys)

            if missing_in_file:
                 log_warning(f"Keys missing in state file '{filename}' (reset to default): {missing_in_file}")
            if extra_in_file:
                 log_warning(f"Extra keys found in state file '{filename}' (ignored): {extra_in_file}")


            log_info(f"Position state loaded from {filename}. Updated {updated_count} fields.")
            display_position_status(position, price_precision_digits, amount_precision_digits) # Display loaded state
        else:
            log_info(f"No position state file found at {filename}. Starting with default state.")
    except json.JSONDecodeError as e:
        log_error(f"Error decoding JSON from state file {filename}: {e}. Starting with default state.", exc_info=True)
        position = default_position_state # Reset to default
    except Exception as e:
        log_error(f"Error loading position state from {filename}: {e}. Starting with default state.", exc_info=True)
        position = default_position_state # Reset to default


# --- Data Fetching Function (Decorated) ---
@api_retry_decorator
def fetch_ohlcv_data(exchange_instance: ccxt.Exchange, trading_symbol: str, tf: str, limit_count: int) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data and returns it as a pandas DataFrame. Retries on network errors."""
    log_debug(f"Fetching {limit_count} candles for {trading_symbol} on {tf} timeframe...")
    try:
        # Check if the market exists locally before fetching
        if trading_symbol not in exchange_instance.markets:
             log_error(f"Symbol {trading_symbol} not loaded in exchange markets.")
             return None

        ohlcv: List[list] = exchange_instance.fetch_ohlcv(trading_symbol, tf, limit=limit_count)
        if not ohlcv:
            log_warning(f"No OHLCV data returned for {trading_symbol} ({tf}). Exchange might be down or symbol inactive.")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        numeric_cols: List[str] = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            # Use errors='coerce' to turn invalid parsing into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_rows: int = len(df)
        # Drop rows where any of the essential numeric columns are NaN
        df.dropna(subset=numeric_cols, inplace=True)
        rows_dropped: int = initial_rows - len(df)
        if rows_dropped > 0:
            log_debug(f"Dropped {rows_dropped} rows with invalid OHLCV data.")

        if df.empty:
            log_warning(f"DataFrame became empty after cleaning for {trading_symbol} ({tf}).")
            return None

        log_debug(f"Successfully fetched and processed {len(df)} candles for {trading_symbol}.")
        return df

    # Keep handling for non-retryable CCXT errors and general exceptions
    except ccxt.BadSymbol as e: # Non-retryable symbol issue
         log_error(f"BadSymbol error fetching OHLCV for {trading_symbol}: {e}. Check symbol format/availability.")
         return None
    except ccxt.ExchangeError as e: # Other exchange-specific errors
        log_error(f"Exchange specific error fetching OHLCV for {trading_symbol}: {e}")
        return None
    except Exception as e:
        # Error already logged by retry decorator if it was a retryable error that failed
        # Log here only if it's an unexpected non-CCXT error within this function's logic
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"An unexpected non-retryable error occurred fetching OHLCV: {e}", exc_info=True)
        return None # Return None if any exception occurred


# --- Enhanced Order Block Identification Function ---
def identify_potential_order_block(df: pd.DataFrame, vol_thresh_mult: float, lookback_len: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Identifies the most recent potential bullish and bearish order blocks based on
    a bearish/bullish candle followed by a high-volume reversal candle that sweeps liquidity.
    Uses config parameters for volume threshold multiplier and lookback length.
    """
    if df is None or df.empty or 'volume' not in df.columns or len(df) < lookback_len + 2:
        log_warning(f"Not enough data or missing volume column for OB detection (need > {lookback_len + 1} rows with volume, got {len(df)}).")
        return None, None

    bullish_ob: Optional[Dict[str, Any]] = None
    bearish_ob: Optional[Dict[str, Any]] = None

    try:
        # Calculate average volume excluding the last (potentially incomplete) candle
        completed_candles_df = df.iloc[:-1]
        avg_volume = 0
        # Ensure enough completed candles for rolling mean, use min_periods for robustness
        if len(completed_candles_df) >= 1:
             avg_volume = completed_candles_df['volume'].rolling(window=lookback_len, min_periods=max(1, lookback_len // 2)).mean().iloc[-1]
             if pd.isna(avg_volume): # Handle case where mean is NaN (e.g., all volumes were NaN)
                 avg_volume = 0

        volume_threshold = avg_volume * vol_thresh_mult if avg_volume > 0 else float('inf')
        log_debug(f"OB Analysis | Lookback: {lookback_len}, Avg Vol: {avg_volume:.2f}, Threshold Vol: {volume_threshold:.2f}")

        # Iterate backwards from second-to-last candle to find the most recent OBs
        # Look back up to 'lookback_len' candles for the pattern start
        search_end_index = max(-1, len(df) - lookback_len - 2)
        for i in range(len(df) - 2, search_end_index, -1):
            if i < 1: break # Need prev_candle

            candle = df.iloc[i]         # Potential "trigger" or "reversal" candle
            prev_candle = df.iloc[i-1]  # Potential candle forming the OB zone

            # Check for NaN values in the candles being examined
            if candle.isnull().any() or prev_candle.isnull().any():
                log_debug(f"Skipping OB check at index {i} due to NaN values.")
                continue

            is_high_volume = candle['volume'] > volume_threshold
            is_bullish_reversal = candle['close'] > candle['open']
            is_bearish_reversal = candle['close'] < candle['open']
            prev_is_bearish = prev_candle['close'] < prev_candle['open']
            prev_is_bullish = prev_candle['close'] > prev_candle['open']

            # --- Potential Bullish OB ---
            # Bearish prev_candle + High-volume Bullish reversal sweeping prev_low + closing strong
            if (not bullish_ob and # Find most recent
                prev_is_bearish and is_bullish_reversal and is_high_volume and
                candle['low'] < prev_candle['low'] and candle['close'] > prev_candle['high']): # Sweep and strong close
                bullish_ob = {
                    'high': prev_candle['high'], 'low': prev_candle['low'],
                    'time': prev_candle.name, # Timestamp of the bearish candle
                    'type': 'bullish'
                }
                log_debug(f"Potential Bullish OB found at {prev_candle.name.strftime('%Y-%m-%d %H:%M')} (Trigger: {candle.name.strftime('%H:%M')})")
                if bearish_ob: break # Found both most recent

            # --- Potential Bearish OB ---
            # Bullish prev_candle + High-volume Bearish reversal sweeping prev_high + closing strong
            elif (not bearish_ob and # Find most recent (use elif as one pair can't be both)
                  prev_is_bullish and is_bearish_reversal and is_high_volume and
                  candle['high'] > prev_candle['high'] and candle['close'] < prev_candle['low']): # Sweep and strong close
                bearish_ob = {
                    'high': prev_candle['high'], 'low': prev_candle['low'],
                    'time': prev_candle.name, # Timestamp of the bullish candle
                    'type': 'bearish'
                }
                log_debug(f"Potential Bearish OB found at {prev_candle.name.strftime('%Y-%m-%d %H:%M')} (Trigger: {candle.name.strftime('%H:%M')})")
                if bullish_ob: break # Found both most recent

            # Optimization: If we've found both types, no need to look further back
            if bullish_ob and bearish_ob:
                break

        return bullish_ob, bearish_ob

    except Exception as e:
        log_error(f"Error in order block identification: {e}", exc_info=True)
        return None, None


# --- Indicator Calculation Function (Adds ATR and Volume MA conditionally) ---
def calculate_technical_indicators(df: Optional[pd.DataFrame],
                                  rsi_len: int, stoch_params: Dict, # Pass params explicitly
                                  calc_atr: bool = False, atr_len: int = 14,
                                  calc_vol_ma: bool = False, vol_ma_len: int = 20
                                  ) -> Optional[pd.DataFrame]:
    """Calculates technical indicators (RSI, Stoch, optional ATR, optional Vol MA)."""
    if df is None or df.empty:
        log_warning("Input DataFrame is None or empty for indicator calculation.")
        return None

    log_debug(f"Calculating indicators on DataFrame with {len(df)} rows...")
    original_columns = set(df.columns)
    calculated_indicators = []
    try:
        # Calculate RSI
        df.ta.rsi(length=rsi_len, append=True)
        calculated_indicators.append(f'RSI_{rsi_len}')

        # Calculate Stochastic Oscillator
        df.ta.stoch(k=stoch_params['k'], d=stoch_params['d'], smooth_k=stoch_params['smooth_k'], append=True)
        calculated_indicators.append(f'STOCHk_{stoch_params["k"]}_{stoch_params["d"]}_{stoch_params["smooth_k"]}')
        calculated_indicators.append(f'STOCHd_{stoch_params["k"]}_{stoch_params["d"]}_{stoch_params["smooth_k"]}')


        # Calculate ATR if needed (for ATR SL/TP or Trailing Stop)
        # pandas_ta typically names the raw ATR column 'ATRr_LENGTH'
        atr_col_name_expected = f'ATRr_{atr_len}'
        if calc_atr:
            df.ta.atr(length=atr_len, append=True)
            log_debug(f"ATR ({atr_len}) calculated.")
            calculated_indicators.append(atr_col_name_expected)

        # Calculate Volume MA if needed (for Volume Confirmation)
        vol_ma_col_name_expected = f'VOL_MA_{vol_ma_len}'
        if calc_vol_ma:
            if 'volume' in df.columns:
                # Use min_periods to handle start of series
                df[vol_ma_col_name_expected] = df['volume'].rolling(window=vol_ma_len, min_periods=max(1, vol_ma_len // 2)).mean()
                log_debug(f"Volume MA ({vol_ma_len}) calculated.")
                calculated_indicators.append(vol_ma_col_name_expected)
            else:
                log_warning("Volume column not found, cannot calculate Volume MA.")

        new_columns: List[str] = list(set(df.columns) - original_columns)
        # Verify that expected columns were actually added
        missing_calc = [ind for ind in calculated_indicators if ind not in df.columns] # Check if calculation failed silently
        if missing_calc:
             log_warning(f"Some indicators might not have been calculated properly: {missing_calc}")

        log_debug(f"Indicators calculated. Columns added: {sorted(new_columns)}")

        initial_len: int = len(df)
        # Drop rows with NaN values generated by indicators (only check calculated indicator columns)
        indicator_cols_present = [col for col in calculated_indicators if col in df.columns]
        if indicator_cols_present:
            df.dropna(subset=indicator_cols_present, inplace=True)
        else:
             log_warning("No indicator columns found to drop NaNs from.")

        rows_dropped_nan: int = initial_len - len(df)
        if rows_dropped_nan > 0:
             log_debug(f"Dropped {rows_dropped_nan} rows with NaN values after indicator calculation.")

        if df.empty:
            log_warning("DataFrame became empty after dropping NaN rows from indicators.")
            return None

        log_debug(f"Indicator calculation complete. DataFrame now has {len(df)} rows.")
        return df

    except Exception as e:
        log_error(f"Error calculating technical indicators: {e}", exc_info=True)
        return None

# --- Position Sizing Function (Decorated) ---
@api_retry_decorator
def calculate_position_size(exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, stop_loss_price: float, risk_perc: float) -> Optional[float]:
    """Calculates position size based on risk percentage and stop-loss distance. Retries on network errors."""
    try:
        log_debug(f"Calculating position size: Symbol={trading_symbol}, Price={current_price}, SL={stop_loss_price}, Risk={risk_perc*100}%")
        # Fetch account balance (This call is retried by the decorator)
        balance = exchange_instance.fetch_balance()
        market = exchange_instance.market(trading_symbol)

        # Determine quote currency
        quote_currency = market.get('quote')
        if not quote_currency:
             # Fallback for symbols like BTC/USDT:USDT
             parts = trading_symbol.split('/')
             if len(parts) > 1:
                 quote_currency = parts[1].split(':')[0]
        if not quote_currency:
             log_error(f"Could not determine quote currency for {trading_symbol}.")
             return None
        log_debug(f"Quote currency: {quote_currency}")

        # Find available balance in quote currency (handle variations in balance structure)
        available_balance = 0.0
        # Try common structures first ('free' might be None, so check dict explicitly)
        if isinstance(balance.get('free'), dict) and quote_currency in balance['free']:
            available_balance = float(balance['free'][quote_currency])
        # Check specific currency entry if 'free' wasn't top-level dict or not present
        elif quote_currency in balance and isinstance(balance[quote_currency], dict):
             available_balance = float(balance[quote_currency].get('free', 0.0))
        # Check unified structure (sometimes under 'total', 'used', 'free')
        elif isinstance(balance.get(quote_currency), dict):
             available_balance = float(balance[quote_currency].get('free', 0.0))
        # Check 'info' field for exchange-specific structure (example: Bybit USDT)
        elif exchange_instance.id == 'bybit' and quote_currency == 'USDT' and 'info' in balance and 'result' in balance['info'] and isinstance(balance['info']['result'], list):
             for wallet_info in balance['info']['result']:
                 if isinstance(wallet_info, dict) and wallet_info.get('coin') == 'USDT':
                      # Use 'availableToWithdraw' or 'availableBalance' depending on API version/preference
                      available_balance = float(wallet_info.get('availableToWithdraw', wallet_info.get('availableBalance', 0.0)))
                      break
        # Add other exchange specific checks here if necessary...

        if available_balance <= 0:
            log_error(f"No available {quote_currency} balance ({available_balance}) found for trading. Check balance structure/funds. Balance details: {balance.get(quote_currency, balance.get('info', 'N/A'))}")
            return None
        log_info(f"Available balance ({quote_currency}): {available_balance:.{price_precision_digits}f}")

        # Calculate risk amount
        risk_amount = available_balance * risk_perc
        log_info(f"Risk per trade ({risk_perc*100:.2f}%): {risk_amount:.{price_precision_digits}f} {quote_currency}")

        # Calculate price difference for SL
        price_diff = abs(current_price - stop_loss_price)

        if price_diff <= min_tick / 2: # Use half tick as threshold
            log_error(f"Stop-loss price {stop_loss_price:.{price_precision_digits}f} is too close to current price {current_price:.{price_precision_digits}f} (Diff: {price_diff}). Cannot calculate size.")
            return None
        log_debug(f"Price difference for SL: {price_diff:.{price_precision_digits}f}")

        # Calculate quantity
        quantity = risk_amount / price_diff
        log_debug(f"Calculated raw quantity: {quantity:.{amount_precision_digits+4}f}") # Show more precision initially

        # Adjust for precision and limits
        try:
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, quantity))
            log_debug(f"Quantity adjusted for precision: {quantity_adjusted:.{amount_precision_digits}f}")
        except ccxt.ExchangeError as precision_error:
             log_warning(f"Could not use exchange.amount_to_precision: {precision_error}. Using raw quantity rounded.")
             # Fallback: round manually (less accurate than exchange method)
             quantity_adjusted = round(quantity, amount_precision_digits)


        if quantity_adjusted <= 0:
            log_error(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} is zero or negative.")
            return None

        # Check limits (min/max amount, min cost)
        limits = market.get('limits', {})
        min_amount = limits.get('amount', {}).get('min')
        max_amount = limits.get('amount', {}).get('max')
        min_cost = limits.get('cost', {}).get('min')

        log_debug(f"Market Limits: Min Qty={min_amount}, Max Qty={max_amount}, Min Cost={min_cost}")

        if min_amount is not None and quantity_adjusted < min_amount:
            log_error(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} is below min amount {min_amount}.")
            return None
        if max_amount is not None and quantity_adjusted > max_amount:
            log_warning(f"Calculated quantity {quantity_adjusted:.{amount_precision_digits}f} exceeds max amount {max_amount}. Capping.")
            quantity_adjusted = float(exchange_instance.amount_to_precision(trading_symbol, max_amount))

        estimated_cost = quantity_adjusted * current_price
        if min_cost is not None and estimated_cost < min_cost:
             log_error(f"Estimated order cost ({estimated_cost:.{price_precision_digits}f}) is below min cost {min_cost}.")
             return None

        # Sanity check cost vs balance
        cost_buffer = 0.995 # Use 99.5% of balance as limit for cost check
        if estimated_cost > available_balance * cost_buffer:
             log_error(f"Estimated cost ({estimated_cost:.{price_precision_digits}f}) exceeds {cost_buffer*100:.1f}% of available balance ({available_balance:.{price_precision_digits}f}). Reduce risk % or add funds.")
             return None

        log_info(f"{NEON_GREEN}Position size calculated: {quantity_adjusted:.{amount_precision_digits}f} {market.get('base', '')}{RESET} (Risking ~{risk_amount:.2f} {quote_currency})")
        return quantity_adjusted

    # Handle non-retryable errors specifically
    except ccxt.AuthenticationError as e:
         log_error(f"Authentication error during position size calculation (fetching balance): {e}")
         return None
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange error during position size calculation: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error calculating position size: {e}", exc_info=True)
        # If it was a retryable error, the decorator would have logged and raised it.
        return None


# --- Order Placement Functions (Decorated) ---

# Helper for cancellation with retry
@api_retry_decorator
def cancel_order_with_retry(exchange_instance: ccxt.Exchange, order_id: str, trading_symbol: str):
    """Cancels an order by ID with retry logic."""
    if not order_id:
        log_debug("No order ID provided to cancel.")
        return # Nothing to cancel

    log_info(f"Attempting to cancel order ID: {order_id} for {trading_symbol}...")
    if not SIMULATION_MODE:
        exchange_instance.cancel_order(order_id, trading_symbol)
        log_info(f"Order {order_id} cancellation request sent.")
    else:
        log_warning(f"SIMULATION: Skipped cancelling order {order_id}.")


@api_retry_decorator
def place_market_order(exchange_instance: ccxt.Exchange, trading_symbol: str, side: str, amount: float, reduce_only: bool = False) -> Optional[Dict[str, Any]]:
    """Places a market order ('buy' or 'sell') with retry logic. Optionally reduceOnly."""
    if side not in ['buy', 'sell']:
        log_error(f"Invalid order side: '{side}'.")
        return None
    if amount <= 0:
        log_error(f"Invalid order amount: {amount}.")
        return None

    try:
        market = exchange_instance.market(trading_symbol)
        base_currency: str = market.get('base', '')
        quote_currency: str = market.get('quote', '')
        amount_formatted = float(exchange_instance.amount_to_precision(trading_symbol, amount))

        log_info(f"Attempting to place {side.upper()} market order for {amount_formatted:.{amount_precision_digits}f} {base_currency} {'(ReduceOnly)' if reduce_only else ''}...")

        params = {}
        # Check if exchange explicitly supports reduceOnly for market orders, common for futures/swap
        # Note: ccxt standard is 'reduceOnly' in params, not always in `has`. Check if market is contract type.
        if reduce_only:
             if market.get('swap') or market.get('future') or market.get('contract'):
                  params['reduceOnly'] = True
                  log_debug("Applying 'reduceOnly=True' to market order.")
             else:
                  log_warning("ReduceOnly requested but market type is likely SPOT. Ignoring reduceOnly param for market order.")


        if SIMULATION_MODE:
            log_warning("!!! SIMULATION: Market order placement skipped.")
            # Create a realistic dummy order response
            sim_price = 0.0
            try: # Fetch ticker safely
                 # Fetching ticker can also fail, apply retry decorator? Maybe overkill here.
                 # Let's apply it for robustness.
                 @api_retry_decorator
                 def fetch_ticker_for_sim(exch, sym):
                    return exch.fetch_ticker(sym)
                 ticker = fetch_ticker_for_sim(exchange_instance, trading_symbol)
                 sim_price = ticker['last'] if ticker and 'last' in ticker else 0.0
            except Exception as ticker_e:
                log_warning(f"Could not fetch ticker for simulation price after retries: {ticker_e}")

            order_id = f'sim_market_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
            sim_cost = amount_formatted * sim_price if sim_price > 0 else 0.0
            order = {
                'id': order_id, 'clientOrderId': order_id, 'timestamp': int(time.time() * 1000),
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'closed', # Assume instant fill
                'symbol': trading_symbol, 'type': 'market', 'timeInForce': 'IOC',
                'side': side, 'price': sim_price, 'average': sim_price,
                'amount': amount_formatted, 'filled': amount_formatted, 'remaining': 0.0,
                'cost': sim_cost, 'fee': None, 'info': {'simulated': True, 'reduceOnly': params.get('reduceOnly', False)}
            }
        else:
            # --- LIVE TRADING ---
            log_warning(f"!!! LIVE MODE: Placing real market order {'(ReduceOnly)' if params.get('reduceOnly') else ''}.")
            order = exchange_instance.create_market_order(trading_symbol, side, amount_formatted, params=params)
            # --- END LIVE TRADING ---

        log_info(f"{side.capitalize()} market order request processed {'(Simulated)' if SIMULATION_MODE else ''}.")
        order_id: Optional[str] = order.get('id')
        order_status: Optional[str] = order.get('status')
        # Use 'average' if available (filled price), fallback to 'price' (less reliable for market)
        order_price: Optional[float] = order.get('average', order.get('price'))
        order_filled: Optional[float] = order.get('filled')
        order_cost: Optional[float] = order.get('cost')

        price_str: str = f"{order_price:.{price_precision_digits}f}" if isinstance(order_price, (int, float)) else "N/A"
        filled_str: str = f"{order_filled:.{amount_precision_digits}f}" if isinstance(order_filled, (int, float)) else "N/A"
        cost_str: str = f"{order_cost:.{price_precision_digits}f}" if isinstance(order_cost, (int, float)) else "N/A"

        log_info(f"Order Result | ID: {order_id or 'N/A'}, Status: {order_status or 'N/A'}, Avg Price: {price_str}, "
                    f"Filled: {filled_str} {base_currency}, Cost: {cost_str} {quote_currency}")
        # Add short delay after placing order to allow exchange processing / state update
        time.sleep(1.5) # Increased delay slightly
        return order

    # Handle specific non-retryable errors
    except ccxt.InsufficientFunds as e:
        log_error(f"Insufficient funds for {side} {amount} {trading_symbol}. Error: {e}")
        return None
    except ccxt.OrderNotFound as e: # Can happen if order rejected immediately
        log_error(f"OrderNotFound error placing {side} {amount} {trading_symbol}. Likely rejected immediately. Error: {e}")
        return None
    except ccxt.InvalidOrder as e:
         log_error(f"Invalid market order parameters for {side} {amount} {trading_symbol}: {e}")
         return None
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error placing {side} market order for {trading_symbol}: {e}")
        return None
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error placing {side} market order: {e}", exc_info=True)
        return None


# Note: Applying retry to the *entire* SL/TP function can be complex if one order
# succeeds and the other fails, leading to duplicate orders on retry.
# This version applies retry to the whole function for simplicity.
# Consider applying retry individually to the internal create_order/create_limit_order calls
# for more granular control in a production environment.
@api_retry_decorator
def place_sl_tp_orders(exchange_instance: ccxt.Exchange, trading_symbol: str, position_side: str, quantity: float, sl_price: float, tp_price: float) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Places stop-loss (stop-market preferred) and take-profit (limit) orders.
    Uses reduceOnly. Retries the whole placement process on network errors.
    """
    sl_order: Optional[Dict[str, Any]] = None
    tp_order: Optional[Dict[str, Any]] = None

    if quantity <= 0 or position_side not in ['long', 'short']:
        log_error(f"Invalid input for SL/TP placement: Qty={quantity}, Side='{position_side}'")
        return None, None

    try:
        market = exchange_instance.market(trading_symbol)
        close_side = 'sell' if position_side == 'long' else 'buy'
        qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, quantity))
        sl_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, sl_price))
        tp_price_formatted = float(exchange_instance.price_to_precision(trading_symbol, tp_price))

        # Check exchange capabilities & Market type
        is_contract_market = market.get('swap') or market.get('future') or market.get('contract')
        has_reduce_only_param = is_contract_market # Assume reduceOnly param usable for contracts

        # Explicit capability checks (less reliable than trying createOrder with params)
        # has_stop_market_method = exchange_instance.has.get('createStopMarketOrder', False)
        # has_stop_limit_method = exchange_instance.has.get('createStopLimitOrder', False)
        has_limit_method = exchange_instance.has.get('createLimitOrder', True) # Assume basic limit exists

        # Check if createOrder likely supports stop types via params (more common approach)
        unified_order_options = exchange_instance.options.get('createOrder', {})
        supported_stop_types = unified_order_options.get('stopTypes', [])
        supports_stop_market_via_create = 'stopMarket' in supported_stop_types
        supports_stop_limit_via_create = 'stopLimit' in supported_stop_types

        # Determine parameters for SL and TP
        sl_params = {'reduceOnly': True} if has_reduce_only_param else {}
        tp_params = {'reduceOnly': True} if has_reduce_only_param else {}

        # --- Place Stop-Loss Order ---
        log_info(f"Attempting SL order: {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} @ trigger ~{sl_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in sl_params else ''}")
        sl_order_type = None
        sl_order_price = None # Only needed for limit types

        # Prefer stopMarket via createOrder if supported
        if supports_stop_market_via_create:
            sl_order_type = 'stopMarket'
            sl_params['stopPrice'] = sl_price_formatted
            log_debug("Using stopMarket type for SL via createOrder.")
        # Fallback to stopLimit via createOrder
        elif supports_stop_limit_via_create:
            sl_order_type = 'stopLimit'
            # StopLimit requires a trigger (stopPrice) and a limit price.
            # Set limit price slightly worse than trigger to increase fill chance
            limit_offset = abs(tp_price_formatted - sl_price_formatted) * 0.1 # 10% of TP-SL range as offset
            limit_offset = max(limit_offset, min_tick * 5) # Ensure offset is at least a few ticks
            sl_limit_price = sl_price_formatted - limit_offset if close_side == 'sell' else sl_price_formatted + limit_offset
            sl_order_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price))

            sl_params['stopPrice'] = sl_price_formatted
            sl_params['price'] = sl_order_price # The limit price for the order placed after trigger
            log_warning(f"Using stopLimit type for SL via createOrder. Trigger: {sl_price_formatted:.{price_precision_digits}f}, Limit: {sl_order_price:.{price_precision_digits}f}.")
        else:
            log_error(f"Exchange {exchange_instance.id} does not seem to support stopMarket or stopLimit orders via CCXT createOrder params. Cannot place automated SL.")
            # Continue to try placing TP

        if sl_order_type:
            if SIMULATION_MODE:
                log_warning("!!! SIMULATION: Stop-loss order placement skipped.")
                sl_order_id = f'sim_sl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                sl_order = {
                    'id': sl_order_id, 'status': 'open', 'symbol': trading_symbol,
                    'type': sl_order_type, 'side': close_side, 'amount': qty_formatted,
                    'price': sl_order_price, # Limit price if stopLimit
                    'stopPrice': sl_params.get('stopPrice'), # Trigger price
                    'average': None, 'filled': 0.0, 'remaining': qty_formatted, 'cost': 0.0,
                    'info': {'simulated': True, 'reduceOnly': sl_params.get('reduceOnly', False)}
                }
            else:
                # --- LIVE TRADING ---
                log_warning(f"!!! LIVE MODE: Placing real {sl_order_type} stop-loss order.")
                # Use create_order as it handles various stop types with params
                sl_order = exchange_instance.create_order(
                    symbol=trading_symbol,
                    type=sl_order_type,
                    side=close_side,
                    amount=qty_formatted,
                    price=sl_order_price, # Required only for limit types (like stopLimit)
                    params=sl_params
                )
                log_info(f"Stop-loss order request processed. ID: {sl_order.get('id', 'N/A')}")
                time.sleep(0.75) # Small delay after live order
                # --- END LIVE TRADING ---

        # --- Place Take-Profit Order ---
        log_info(f"Attempting TP order: {close_side.upper()} {qty_formatted:.{amount_precision_digits}f} @ limit {tp_price_formatted:.{price_precision_digits}f} {'(ReduceOnly)' if 'reduceOnly' in tp_params else ''}")
        tp_order_type = 'limit' # Standard limit order for TP

        if has_limit_method: # Check if limit orders are supported (should almost always be true)
             if SIMULATION_MODE:
                 log_warning("!!! SIMULATION: Take-profit order placement skipped.")
                 tp_order_id = f'sim_tp_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                 tp_order = {
                     'id': tp_order_id, 'status': 'open', 'symbol': trading_symbol,
                     'type': tp_order_type, 'side': close_side, 'amount': qty_formatted,
                     'price': tp_price_formatted, 'average': None,
                     'filled': 0.0, 'remaining': qty_formatted, 'cost': 0.0,
                     'info': {'simulated': True, 'reduceOnly': tp_params.get('reduceOnly', False)}
                 }
             else:
                # --- LIVE TRADING ---
                 log_warning(f"!!! LIVE MODE: Placing real {tp_order_type} take-profit order.")
                 tp_order = exchange_instance.create_limit_order(
                     symbol=trading_symbol,
                     side=close_side,
                     amount=qty_formatted,
                     price=tp_price_formatted,
                     params=tp_params
                 )
                 log_info(f"Take-profit order request processed. ID: {tp_order.get('id', 'N/A')}")
                 time.sleep(0.75) # Small delay after live order
                 # --- END LIVE TRADING ---
        else:
             log_error(f"Exchange {exchange_instance.id} does not support limit orders. Cannot place take-profit.")
             tp_order = None


        # Final check and warnings if one failed
        sl_ok = sl_order and sl_order.get('id')
        tp_ok = tp_order and tp_order.get('id')
        if sl_ok and not tp_ok: log_warning("SL placed, but TP failed.")
        elif not sl_ok and tp_ok: log_warning("TP placed, but SL failed. Position unprotected!")
        elif not sl_ok and not tp_ok: log_error("Both SL and TP order placements failed.")

        return sl_order, tp_order

    # Handle specific non-retryable errors
    except ccxt.InvalidOrder as e:
        log_error(f"Invalid order parameters placing SL/TP: {e}")
        # Return whatever might have succeeded before the error
        return sl_order, tp_order
    except ccxt.InsufficientFunds as e: # Should not happen with reduceOnly, but possible
         log_error(f"Insufficient funds error during SL/TP placement (unexpected): {e}")
         return sl_order, tp_order
    except ccxt.ExchangeError as e: # Catch other non-retryable exchange errors
        log_error(f"Exchange specific error placing SL/TP orders: {e}")
        return sl_order, tp_order
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error in place_sl_tp_orders: {e}", exc_info=True)
        # Return None, None to indicate total failure if an exception occurred
        return None, None


# --- Position and Order Check Function (Decorated) ---
@api_retry_decorator
def check_position_and_orders(exchange_instance: ccxt.Exchange, trading_symbol: str) -> bool:
    """
    Checks if tracked SL/TP orders are still open. If not, assumes filled,
    cancels the other order, resets local state, and returns True. Retries on network errors.
    Uses fetch_open_orders.
    """
    global position
    if position['status'] is None:
        return False # No active position to check

    log_debug(f"Checking open orders vs local state for {trading_symbol}...")
    position_reset_flag = False
    try:
        # Fetch Open Orders for the specific symbol (retried by decorator)
        open_orders = exchange_instance.fetch_open_orders(trading_symbol)
        log_debug(f"Found {len(open_orders)} open orders for {trading_symbol}.")

        sl_order_id = position.get('sl_order_id')
        tp_order_id = position.get('tp_order_id')

        if not sl_order_id and not tp_order_id:
             # This state can occur if SL/TP placement failed initially or after a restart
             # without proper state restoration of order IDs.
             log_warning(f"In {position['status']} position but no SL/TP order IDs tracked. Cannot verify closure via orders.")
             # Alternative: Try fetching current position size from exchange here?
             # `fetch_positions` is often needed for futures/swaps, but can be complex/inconsistent.
             # Sticking to order-based check for now.
             # TODO: Consider adding a fetch_positions check as a fallback if no orders are tracked.
             return False # Cannot determine closure reliably

        # Check if tracked orders are still present in the fetched open orders
        open_order_ids = {order.get('id') for order in open_orders if order.get('id')}
        sl_found_open = sl_order_id in open_order_ids
        tp_found_open = tp_order_id in open_order_ids

        log_debug(f"Tracked SL ({sl_order_id or 'None'}) open: {sl_found_open}. Tracked TP ({tp_order_id or 'None'}) open: {tp_found_open}.")

        # --- Logic for State Reset ---
        order_to_cancel_id = None
        assumed_close_reason = None

        # If SL was tracked but is no longer open:
        if sl_order_id and not sl_found_open:
            log_info(f"{NEON_YELLOW}Stop-loss order {sl_order_id} no longer open. Assuming position closed via SL.{RESET}")
            position_reset_flag = True
            order_to_cancel_id = tp_order_id # Attempt to cancel the TP order
            assumed_close_reason = "SL"

        # Else if TP was tracked but is no longer open:
        elif tp_order_id and not tp_found_open:
            log_info(f"{NEON_GREEN}Take-profit order {tp_order_id} no longer open. Assuming position closed via TP.{RESET}")
            position_reset_flag = True
            order_to_cancel_id = sl_order_id # Attempt to cancel the SL order
            assumed_close_reason = "TP"

        # If a reset was triggered:
        if position_reset_flag:
            # Attempt to cancel the other order using the retry helper
            if order_to_cancel_id:
                 log_info(f"Attempting to cancel leftover {'TP' if assumed_close_reason == 'SL' else 'SL'} order {order_to_cancel_id}...")
                 try:
                     # Use the decorated cancel helper
                     cancel_order_with_retry(exchange_instance, order_to_cancel_id, trading_symbol)
                 except ccxt.OrderNotFound:
                     log_info(f"Order {order_to_cancel_id} was already closed/cancelled.")
                 except Exception as e:
                     # Error logged by cancel_order_with_retry or its decorator
                     log_error(f"Failed to cancel leftover order {order_to_cancel_id} after retries. Manual check advised.", exc_info=False) # Don't need full trace here
            else:
                 log_debug(f"No corresponding {'TP' if assumed_close_reason == 'SL' else 'SL'} order ID was tracked or provided to cancel.")


            log_info("Resetting local position state.")
            # Reset all relevant fields using the default structure
            default_state = {k: None for k in position.keys()}
            position.update(default_state) # More robust reset
            save_position_state() # Save the reset state
            display_position_status(position, price_precision_digits, amount_precision_digits) # Show updated status
            return True # Indicate position was reset

        # If both orders are still open, or only one was tracked and it's still open
        log_debug("Position state appears consistent with tracked open orders.")
        return False # Position was not reset

    # Handle specific non-retryable errors from fetch_open_orders if they bypass decorator
    except ccxt.AuthenticationError as e:
         log_error(f"Authentication error checking position/orders: {e}")
         return False
    except ccxt.ExchangeError as e:
        log_error(f"Exchange error checking position/orders: {e}")
        return False
    except Exception as e:
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             log_error(f"Unexpected non-retryable error checking position/orders: {e}", exc_info=True)
        # Let retry handler manage retryable exceptions
        return False # Assume state is uncertain if error occurred

# --- Trailing Stop Loss Update Function (Adapted & Simplified Placement) ---
def update_trailing_stop(exchange_instance: ccxt.Exchange, trading_symbol: str, current_price: float, last_atr: float):
    """Checks and updates the trailing stop loss if conditions are met."""
    global position
    # Pre-checks
    if position['status'] is None: return
    if not enable_trailing_stop: return
    if position['sl_order_id'] is None:
        if enable_trailing_stop: # Log only if TSL is enabled but no SL ID exists
            log_warning("Cannot perform trailing stop check: No active SL order ID tracked.")
        return
    if last_atr <= 0:
        log_warning("Cannot perform trailing stop check: Invalid ATR value.")
        return

    log_debug(f"Checking trailing stop. Current SL Order: {position['sl_order_id']}, Current Tracked TSL Price: {position.get('current_trailing_sl_price')}")

    new_potential_tsl = None
    activation_threshold_met = False
    # Use current_trailing_sl_price if set, otherwise use the initial stop_loss from state
    current_effective_sl = position.get('current_trailing_sl_price') or position.get('stop_loss')

    if current_effective_sl is None:
        log_warning("Cannot check TSL improvement: current effective SL price is None.")
        return

    # --- Update Peak Prices & Check Activation ---
    entry_price = position.get('entry_price', current_price) # Fallback

    if position['status'] == 'long':
        highest_seen = position.get('highest_price_since_entry', entry_price)
        if current_price > highest_seen:
            position['highest_price_since_entry'] = current_price
            highest_seen = current_price
            # save_position_state() # Persist the new high - potentially too frequent

        activation_price = entry_price + (last_atr * trailing_stop_activation_atr_multiplier)
        if highest_seen > activation_price:
            activation_threshold_met = True
            new_potential_tsl = highest_seen - (last_atr * trailing_stop_atr_multiplier)
            log_debug(f"Long TSL Activation Met. Highest: {highest_seen:.{price_precision_digits}f} > Activation: {activation_price:.{price_precision_digits}f}. Potential New TSL: {new_potential_tsl:.{price_precision_digits}f}")
        else:
             log_debug(f"Long TSL Activation NOT Met. Highest: {highest_seen:.{price_precision_digits}f} <= Activation: {activation_price:.{price_precision_digits}f}")

    elif position['status'] == 'short':
        lowest_seen = position.get('lowest_price_since_entry', entry_price)
        if current_price < lowest_seen:
            position['lowest_price_since_entry'] = current_price
            lowest_seen = current_price
            # save_position_state() # Persist the new low - potentially too frequent

        activation_price = entry_price - (last_atr * trailing_stop_activation_atr_multiplier)
        if lowest_seen < activation_price:
            activation_threshold_met = True
            new_potential_tsl = lowest_seen + (last_atr * trailing_stop_atr_multiplier)
            log_debug(f"Short TSL Activation Met. Lowest: {lowest_seen:.{price_precision_digits}f} < Activation: {activation_price:.{price_precision_digits}f}. Potential New TSL: {new_potential_tsl:.{price_precision_digits}f}")
        else:
            log_debug(f"Short TSL Activation NOT Met. Lowest: {lowest_seen:.{price_precision_digits}f} >= Activation: {activation_price:.{price_precision_digits}f}")

    # --- Check Improvement & Update ---
    if activation_threshold_met and new_potential_tsl is not None:
        should_update_tsl = False
        new_tsl_formatted = float(exchange_instance.price_to_precision(trading_symbol, new_potential_tsl))

        # Check if new TSL is better than current effective SL and valid relative to current price
        if position['status'] == 'long':
             if new_tsl_formatted > current_effective_sl and new_tsl_formatted < current_price:
                 should_update_tsl = True
                 log_debug(f"Long TSL Improvement Check -> OK")
             else:
                 log_debug(f"Long TSL Improvement Check -> NO UPDATE")
        elif position['status'] == 'short':
             if new_tsl_formatted < current_effective_sl and new_tsl_formatted > current_price:
                 should_update_tsl = True
                 log_debug(f"Short TSL Improvement Check -> OK")
             else:
                  log_debug(f"Short TSL Improvement Check -> NO UPDATE")

        # --- Execute TSL Update if needed ---
        if should_update_tsl:
            log_info(f"{NEON_YELLOW}Trailing Stop Update Triggered! New target SL: {new_tsl_formatted:.{price_precision_digits}f}{RESET}")
            old_sl_id = position['sl_order_id'] # Already confirmed not None above

            try:
                # 1. Cancel Old SL Order (using retry helper)
                log_info(f"Cancelling old SL order {old_sl_id} for TSL update...")
                cancel_order_with_retry(exchange_instance, old_sl_id, trading_symbol)
                log_info(f"Old SL order {old_sl_id} cancellation request sent/simulated.")
                time.sleep(1.0) # Allow cancellation processing

                # 2. Place New SL Order (Direct placement logic adapted from place_sl_tp_orders)
                log_info(f"Placing new TSL order at {new_tsl_formatted:.{price_precision_digits}f}")
                new_sl_order_result: Optional[Dict[str, Any]] = None
                try:
                    close_side = 'sell' if position['status'] == 'long' else 'buy'
                    qty_formatted = float(exchange_instance.amount_to_precision(trading_symbol, position['quantity']))
                    market = exchange_instance.market(trading_symbol)
                    is_contract_market = market.get('swap') or market.get('future') or market.get('contract')
                    has_reduce_only_param = is_contract_market

                    new_sl_params = {'reduceOnly': True} if has_reduce_only_param else {}
                    new_sl_order_type = None
                    new_sl_order_price = None # For stopLimit

                    # Check preferred stop types via createOrder params
                    unified_order_options = exchange_instance.options.get('createOrder', {})
                    supported_stop_types = unified_order_options.get('stopTypes', [])
                    supports_stop_market_via_create = 'stopMarket' in supported_stop_types
                    supports_stop_limit_via_create = 'stopLimit' in supported_stop_types

                    if supports_stop_market_via_create:
                        new_sl_order_type = 'stopMarket'
                        new_sl_params['stopPrice'] = new_tsl_formatted
                    elif supports_stop_limit_via_create:
                        new_sl_order_type = 'stopLimit'
                        original_tp_price = position.get('take_profit', current_price * (1.05 if position['status'] == 'long' else 0.95)) # Fallback TP
                        limit_offset = abs(original_tp_price - new_tsl_formatted) * 0.1
                        limit_offset = max(limit_offset, min_tick * 5)
                        sl_limit_price = new_tsl_formatted - limit_offset if close_side == 'sell' else new_tsl_formatted + limit_offset
                        new_sl_order_price = float(exchange_instance.price_to_precision(trading_symbol, sl_limit_price))
                        new_sl_params['stopPrice'] = new_tsl_formatted
                        new_sl_params['price'] = new_sl_order_price
                    else:
                         raise ccxt.NotSupported(f"Exchange {exchange_instance.id} supports neither stopMarket nor stopLimit via createOrder for TSL.")

                    log_info(f"Attempting direct placement of new TSL order ({new_sl_order_type})")
                    if SIMULATION_MODE:
                        log_warning("!!! SIMULATION: New TSL order placement skipped.")
                        new_sl_order_id = f'sim_tsl_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S%f")}'
                        new_sl_order_result = {'id': new_sl_order_id, 'status': 'open', 'info': {'simulated': True}}
                    else:
                        log_warning(f"!!! LIVE MODE: Placing real {new_sl_order_type} TSL order.")
                        # Apply retry decorator specifically to this placement call
                        @api_retry_decorator
                        def place_new_tsl_order_live():
                            return exchange_instance.create_order(
                                symbol=trading_symbol, type=new_sl_order_type, side=close_side,
                                amount=qty_formatted, price=new_sl_order_price, params=new_sl_params
                            )
                        new_sl_order_result = place_new_tsl_order_live()

                    # --- Update State on Successful Placement ---
                    if new_sl_order_result and new_sl_order_result.get('id'):
                        new_id = new_sl_order_result['id']
                        log_info(f"New trailing SL order placed successfully: ID {new_id}")
                        position['stop_loss'] = new_tsl_formatted # Update the reference SL price
                        position['sl_order_id'] = new_id
                        position['current_trailing_sl_price'] = new_tsl_formatted # Mark the active TSL price
                        save_position_state()
                    else:
                        log_error(f"Failed to place new trailing SL order after cancelling old one {old_sl_id}. POSITION MAY BE UNPROTECTED.")
                        position['sl_order_id'] = None # Mark SL as lost
                        position['current_trailing_sl_price'] = None
                        save_position_state()

                except Exception as place_e:
                    # Error logged by retry decorator or generic handler if non-retryable
                    log_error(f"Error placing new trailing SL order: {place_e}. POSITION MAY BE UNPROTECTED.", exc_info=True)
                    position['sl_order_id'] = None
                    position['current_trailing_sl_price'] = None
                    save_position_state()

            except ccxt.OrderNotFound:
                log_warning(f"Old SL order {old_sl_id} not found during TSL update cancellation (might have been filled/cancelled already).")
                # If SL filled, check_position_and_orders should handle the reset next cycle.
                # Mark TSL as inactive if SL ID is now gone.
                position['current_trailing_sl_price'] = None
                position['sl_order_id'] = None # Ensure SL ID is cleared
                save_position_state()
            except Exception as cancel_e:
                # Error logged by cancel_order_with_retry or its decorator
                log_error(f"Error cancelling old SL order {old_sl_id} during TSL update: {cancel_e}. Aborting TSL placement.", exc_info=False)
                # Do not proceed to place new SL if cancellation failed unexpectedly.

# --- Main Trading Loop ---
log_info(f"Initializing trading bot for {symbol} on {timeframe}...")
# Load position state ONCE at startup
load_position_state()
log_info(f"Risk per trade: {risk_percentage*100:.2f}%")
log_info(f"Bot check interval: {sleep_interval_seconds} seconds ({sleep_interval_seconds/60:.1f} minutes)")
log_info(f"{NEON_YELLOW}Press Ctrl+C to stop the bot gracefully.{RESET}")

while True:
    try:
        cycle_start_time: pd.Timestamp = pd.Timestamp.now(tz='UTC')
        print_cycle_divider(cycle_start_time)

        # 1. Check Position/Order Consistency FIRST
        position_was_reset = check_position_and_orders(exchange, symbol)
        display_position_status(position, price_precision_digits, amount_precision_digits) # Display status after check
        if position_was_reset:
             log_info("Position state reset by order check. Proceeding to check for new entries.")
             # Continue loop to immediately check for signals without sleeping

        # 2. Fetch Fresh OHLCV Data
        ohlcv_df: Optional[pd.DataFrame] = fetch_ohlcv_data(exchange, symbol, timeframe, limit_count=data_limit)
        if ohlcv_df is None or ohlcv_df.empty:
            log_warning(f"Could not fetch valid OHLCV data. Waiting...")
            neon_sleep_timer(sleep_interval_seconds)
            continue

        # 3. Calculate Technical Indicators (Conditionally include ATR, Vol MA)
        needs_atr = enable_atr_sl_tp or enable_trailing_stop
        needs_vol_ma = entry_volume_confirmation_enabled
        stoch_params = {'k': stoch_k, 'd': stoch_d, 'smooth_k': stoch_smooth_k}
        df_with_indicators: Optional[pd.DataFrame] = calculate_technical_indicators(
            ohlcv_df.copy(), # Use copy
            rsi_len=rsi_length, stoch_params=stoch_params,
            calc_atr=needs_atr, atr_len=atr_length,
            calc_vol_ma=needs_vol_ma, vol_ma_len=entry_volume_ma_length
        )
        if df_with_indicators is None or df_with_indicators.empty:
             log_warning(f"Indicator calculation failed. Waiting...")
             neon_sleep_timer(sleep_interval_seconds)
             continue

        # 4. Get Latest Data and Indicator Values
        if len(df_with_indicators) < 2:
            log_warning("Not enough data points after indicator calculation. Waiting...")
            neon_sleep_timer(sleep_interval_seconds)
            continue

        latest_data: pd.Series = df_with_indicators.iloc[-1]
        # Construct indicator column names
        rsi_col_name: str = f'RSI_{rsi_length}'
        stoch_k_col_name: str = f'STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth_k}'
        stoch_d_col_name: str = f'STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth_k}'
        # ATR column from pandas_ta is typically ATRr_LENGTH (raw ATR)
        atr_col_name: Optional[str] = f'ATRr_{atr_length}' if needs_atr else None
        vol_ma_col_name: Optional[str] = f'VOL_MA_{entry_volume_ma_length}' if needs_vol_ma else None

        # Check required columns exist
        required_base_cols: List[str] = ['close', 'high', 'low', 'open', 'volume']
        required_indicator_cols: List[str] = [rsi_col_name, stoch_k_col_name, stoch_d_col_name]
        if needs_atr and atr_col_name: required_indicator_cols.append(atr_col_name)
        if needs_vol_ma and vol_ma_col_name: required_indicator_cols.append(vol_ma_col_name)

        all_required_cols = required_base_cols + [col for col in required_indicator_cols if col is not None]
        missing_cols = [col for col in all_required_cols if col not in df_with_indicators.columns]

        if missing_cols:
            log_error(f"Required columns missing in DataFrame: {missing_cols}. Check config/data. Available: {df_with_indicators.columns.tolist()}")
            neon_sleep_timer(sleep_interval_seconds)
            continue

        # Extract latest values safely, checking for NaNs
        try:
            current_price: float = float(latest_data['close'])
            current_high: float = float(latest_data['high'])
            current_low: float = float(latest_data['low'])
            last_rsi: float = float(latest_data[rsi_col_name])
            last_stoch_k: float = float(latest_data[stoch_k_col_name])
            last_stoch_d: float = float(latest_data[stoch_d_col_name])

            last_atr: Optional[float] = None
            if needs_atr and atr_col_name:
                atr_val = latest_data.get(atr_col_name)
                if pd.notna(atr_val):
                    last_atr = float(atr_val)
                else:
                     log_warning(f"ATR value is NaN for the latest candle. ATR-based logic will be skipped.")


            current_volume: Optional[float] = None
            last_volume_ma: Optional[float] = None
            if needs_vol_ma and vol_ma_col_name and 'volume' in latest_data:
                 vol_val = latest_data.get('volume')
                 vol_ma_val = latest_data.get(vol_ma_col_name)
                 if pd.notna(vol_val): current_volume = float(vol_val)
                 if pd.notna(vol_ma_val): last_volume_ma = float(vol_ma_val)
                 if current_volume is None or last_volume_ma is None:
                     log_warning("Volume or Volume MA is NaN. Volume confirmation may be skipped.")

            # Final check for essential NaNs
            essential_values = [current_price, last_rsi, last_stoch_k, last_stoch_d]
            if any(pd.isna(v) for v in essential_values):
                 raise ValueError("Essential indicator value is NaN.")

        except (KeyError, ValueError, TypeError) as e:
             log_error(f"Error extracting latest data or NaN found: {e}. Data: {latest_data.to_dict()}", exc_info=True)
             neon_sleep_timer(sleep_interval_seconds)
             continue

        # Display Market Stats
        display_market_stats(current_price, last_rsi, last_stoch_k, last_stoch_d, last_atr, price_precision_digits)

        # 5. Identify Order Blocks
        bullish_ob, bearish_ob = identify_potential_order_block(
            df_with_indicators,
            vol_thresh_mult=ob_volume_threshold_multiplier,
            lookback_len=ob_lookback
        )
        display_order_blocks(bullish_ob, bearish_ob, price_precision_digits)


        # 6. Apply Trading Logic

        # --- Check if ALREADY IN A POSITION ---
        if position['status'] is not None:
            log_info(f"Currently in {position['status'].upper()} position.")

            # A. Check Trailing Stop Logic (only if enabled and ATR available)
            # Note: update_trailing_stop now handles internal checks for ATR validity etc.
            update_trailing_stop(exchange, symbol, current_price, last_atr if last_atr else 0.0)


            # B. Check for Indicator-Based Exits (Optional Secondary Exit)
            # This acts as a fallback IF SL/TP somehow fail or if you want faster exits.
            # Ensure this doesn't trigger immediately after entry if conditions are still met.
            # Add a check to ensure we are not on the entry candle itself (optional).
            execute_indicator_exit = False
            exit_reason = ""
            if position['status'] == 'long' and last_rsi > rsi_overbought:
                 exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} > {rsi_overbought})"
                 execute_indicator_exit = True
            elif position['status'] == 'short' and last_rsi < rsi_oversold:
                 exit_reason = f"Indicator Exit Signal (RSI {last_rsi:.1f} < {rsi_oversold})"
                 execute_indicator_exit = True

            if execute_indicator_exit:
                display_signal("Exit", position['status'], exit_reason)
                log_warning(f"Attempting to close {position['status'].upper()} position via FALLBACK market order due to indicator signal.")

                # IMPORTANT: Cancel existing SL and TP orders *before* sending market order
                sl_id_to_cancel = position.get('sl_order_id')
                tp_id_to_cancel = position.get('tp_order_id')
                orders_cancelled_successfully = True # Assume success initially

                for order_id, order_type in [(sl_id_to_cancel, "SL"), (tp_id_to_cancel, "TP")]:
                    if order_id:
                        log_info(f"Cancelling existing {order_type} order {order_id} before fallback exit...")
                        try:
                            cancel_order_with_retry(exchange, order_id, symbol)
                        except ccxt.OrderNotFound:
                            log_info(f"{order_type} order {order_id} already closed/cancelled.")
                        except Exception as e:
                            log_error(f"Error cancelling {order_type} order {order_id} during fallback exit: {e}", exc_info=False)
                            orders_cancelled_successfully = False # Mark cancellation as potentially failed

                if not orders_cancelled_successfully:
                     log_error("Failed to cancel one or both SL/TP orders. Aborting fallback market exit to avoid potential issues. MANUAL INTERVENTION MAY BE REQUIRED.")
                else:
                    # Proceed with market exit order (use reduceOnly=True)
                    close_side = 'sell' if position['status'] == 'long' else 'buy'
                    exit_qty = position.get('quantity') # Get quantity from state
                    if exit_qty is None or exit_qty <= 0:
                         log_error("Cannot place fallback exit order: Invalid quantity in position state.")
                    else:
                        order_result = place_market_order(exchange, symbol, close_side, exit_qty, reduce_only=True)

                        # Check if order likely filled (can be tricky with market orders)
                        # We reset state assuming it worked, check_position_and_orders will confirm next cycle
                        if order_result and order_result.get('id'):
                            log_info(f"Fallback {position['status']} position close order placed: ID {order_result.get('id', 'N/A')}")
                        else:
                            log_error(f"Fallback market order placement FAILED for {position['status']} position.")
                            # Critical: Bot tried to exit but failed. Manual intervention likely needed.

                        # Reset position state immediately after attempting market exit
                        # regardless of confirmation, as intent was to close.
                        log_info("Resetting local position state after indicator-based market exit attempt.")
                        default_state = {k: None for k in position.keys()}
                        position.update(default_state) # More robust reset
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits)
                        # Exit loop for this cycle after attempting exit
                        neon_sleep_timer(sleep_interval_seconds)
                        continue # Go to next cycle immediately after exit attempt
            else:
                 # If no indicator exit, just log monitoring status
                 log_info(f"Monitoring {position['status'].upper()} position. Waiting for SL/TP ({position.get('sl_order_id') or 'N/A'}/{position.get('tp_order_id') or 'N/A'}) or TSL update.")


        # --- Check for NEW ENTRIES (only if not currently in a position) ---
        else: # position['status'] is None
            log_info("No active position. Checking for entry signals...")

            # --- Volume Confirmation Check ---
            volume_confirmed = False
            if entry_volume_confirmation_enabled:
                if current_volume is not None and last_volume_ma is not None and last_volume_ma > 0:
                    if current_volume > (last_volume_ma * entry_volume_multiplier):
                        volume_confirmed = True
                        log_debug(f"Volume confirmed: Current Vol ({current_volume:.2f}) > MA Vol ({last_volume_ma:.2f}) * {entry_volume_multiplier}")
                    else:
                        log_debug(f"Volume NOT confirmed: Current Vol ({current_volume:.2f}), MA Vol ({last_volume_ma:.2f}), Threshold ({last_volume_ma * entry_volume_multiplier:.2f})")
                else:
                    log_debug("Volume confirmation enabled but volume or MA data is missing/invalid.")
            else:
                volume_confirmed = True # Skip check if disabled

            # --- Base Signal Conditions ---
            base_long_signal = last_rsi < rsi_oversold and last_stoch_k < stoch_oversold
            base_short_signal = last_rsi > rsi_overbought and last_stoch_k > stoch_overbought

            # --- OB Price Check ---
            long_ob_price_check = False
            long_ob_reason_part = ""
            if base_long_signal and bullish_ob:
                ob_range = bullish_ob['high'] - bullish_ob['low']
                # Allow entry if price is within OB or up to 10% of OB range above high
                entry_zone_high = bullish_ob['high'] + max(ob_range * 0.10, min_tick * 2) # Add min tick buffer
                entry_zone_low = bullish_ob['low']
                if entry_zone_low <= current_price <= entry_zone_high:
                    long_ob_price_check = True
                    long_ob_reason_part = f"Price near Bullish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                else:
                    log_debug(f"Base Long signal met, but price {current_price:.{price_precision_digits}f} outside Bullish OB entry zone.")
            elif base_long_signal:
                 log_debug("Base Long signal met, but no recent Bullish OB found.")

            short_ob_price_check = False
            short_ob_reason_part = ""
            if base_short_signal and bearish_ob:
                ob_range = bearish_ob['high'] - bearish_ob['low']
                # Allow entry if price is within OB or down to 10% of OB range below low
                entry_zone_low = bearish_ob['low'] - max(ob_range * 0.10, min_tick * 2)
                entry_zone_high = bearish_ob['high']
                if entry_zone_low <= current_price <= entry_zone_high:
                    short_ob_price_check = True
                    short_ob_reason_part = f"Price near Bearish OB [{entry_zone_low:.{price_precision_digits}f} - {entry_zone_high:.{price_precision_digits}f}]"
                else:
                     log_debug(f"Base Short signal met, but price {current_price:.{price_precision_digits}f} outside Bearish OB entry zone.")
            elif base_short_signal:
                  log_debug("Base Short signal met, but no recent Bearish OB found.")


            # --- Combine Conditions for Final Entry Signal ---
            long_entry_condition = base_long_signal and long_ob_price_check and volume_confirmed
            short_entry_condition = base_short_signal and short_ob_price_check and volume_confirmed

            long_reason = ""
            short_reason = ""
            if long_entry_condition:
                 long_reason = (f"RSI ({last_rsi:.1f} < {rsi_oversold}), "
                                f"StochK ({last_stoch_k:.1f} < {stoch_oversold}), "
                                f"{long_ob_reason_part}" +
                                (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))
            elif short_entry_condition:
                 short_reason = (f"RSI ({last_rsi:.1f} > {rsi_overbought}), "
                                 f"StochK ({last_stoch_k:.1f} > {stoch_overbought}), "
                                 f"{short_ob_reason_part}" +
                                 (", Volume Confirmed" if entry_volume_confirmation_enabled else ""))
            # Log reasons for failure if base signal was met but other checks failed
            elif base_long_signal and long_ob_price_check and not volume_confirmed:
                 log_debug("Long OB/Price conditions met, but volume not confirmed.")
            elif base_short_signal and short_ob_price_check and not volume_confirmed:
                 log_debug("Short OB/Price conditions met, but volume not confirmed.")


            # --- Execute Entry ---
            if long_entry_condition:
                display_signal("Entry", "long", long_reason)

                # Calculate SL/TP prices
                stop_loss_price = 0.0
                take_profit_price = 0.0
                if enable_atr_sl_tp:
                    if last_atr is None or last_atr <= 0:
                        log_error("Cannot calculate ATR SL/TP: Invalid ATR value. Skipping LONG entry.")
                        continue # Skip to next cycle
                    stop_loss_price = current_price - (last_atr * atr_sl_multiplier)
                    take_profit_price = current_price + (last_atr * atr_tp_multiplier)
                    log_info(f"Calculated ATR-based SL: {stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP: {take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)")
                else: # Fixed percentage
                    stop_loss_price = current_price * (1 - stop_loss_percentage)
                    take_profit_price = current_price * (1 + take_profit_percentage)
                    log_info(f"Calculated Fixed % SL: {stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage*100:.1f}%), TP: {take_profit_price:.{price_precision_digits}f} ({take_profit_percentage*100:.1f}%)")

                # Adjust SL based on Bullish OB low if it provides tighter stop
                # Ensure bullish_ob exists and low is valid
                if bullish_ob and isinstance(bullish_ob.get('low'), (float, int)) and bullish_ob['low'] > stop_loss_price:
                    # Set SL just below the OB low (add buffer relative to OB low)
                    adjusted_sl = bullish_ob['low'] * (1 - 0.0005) # Smaller buffer, e.g., 0.05% below low
                    # Ensure adjustment doesn't push SL too close or above current price, and is still below OB low
                    if adjusted_sl < current_price and adjusted_sl < bullish_ob['low']:
                         stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))
                         log_info(f"Adjusted SL tighter based on Bullish OB low: {stop_loss_price:.{price_precision_digits}f}")
                    else:
                         log_warning(f"Could not adjust SL based on OB low {bullish_ob['low']:.{price_precision_digits}f} as it's too close or invalid.")


                # Final SL/TP validation
                if stop_loss_price >= current_price or take_profit_price <= current_price:
                     log_error(f"Invalid SL/TP calculated: SL {stop_loss_price:.{price_precision_digits}f}, TP {take_profit_price:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping LONG entry.")
                     continue

                # Calculate position size
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)
                if quantity is None or quantity <= 0:
                    log_error("Failed to calculate valid position size. Skipping LONG entry.")
                else:
                    # Place entry market order
                    entry_order_result = place_market_order(exchange, symbol, 'buy', quantity, reduce_only=False) # Entry is not reduceOnly
                    # Check status carefully - 'closed' is ideal, but accept 'open' if filled > 0 as market orders might report open briefly
                    is_filled = entry_order_result and (entry_order_result.get('status') == 'closed' or (entry_order_result.get('status') == 'open' and entry_order_result.get('filled', 0) > 0))

                    if is_filled:
                        entry_price_actual = entry_order_result.get('average', current_price) # Use filled price if available
                        filled_quantity = entry_order_result.get('filled', quantity) # Use filled qty if available

                        log_info(f"Long position entry order filled/processed: ID {entry_order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f} for {filled_quantity:.{amount_precision_digits}f}")

                        # Place SL/TP orders *after* confirming entry
                        sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'long', filled_quantity, stop_loss_price, take_profit_price)

                        # Update position state ONLY if entry was successful
                        position.update({
                            'status': 'long',
                            'entry_price': entry_price_actual,
                            'quantity': filled_quantity,
                            'order_id': entry_order_result.get('id'),
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'entry_time': pd.Timestamp.now(tz='UTC'),
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None,
                            # Initialize TSL fields
                            'highest_price_since_entry': entry_price_actual,
                            'lowest_price_since_entry': None,
                            'current_trailing_sl_price': None
                        })
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits) # Show new status

                        if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):
                            log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                    else:
                        log_error(f"Failed to place or confirm fill for long entry order. Result: {entry_order_result}")

            elif short_entry_condition: # Use elif to prevent long/short in same cycle
                display_signal("Entry", "short", short_reason)

                # Calculate SL/TP prices
                stop_loss_price = 0.0
                take_profit_price = 0.0
                if enable_atr_sl_tp:
                    if last_atr is None or last_atr <= 0:
                        log_error("Cannot calculate ATR SL/TP: Invalid ATR value. Skipping SHORT entry.")
                        continue
                    stop_loss_price = current_price + (last_atr * atr_sl_multiplier)
                    take_profit_price = current_price - (last_atr * atr_tp_multiplier)
                    log_info(f"Calculated ATR-based SL: {stop_loss_price:.{price_precision_digits}f} ({atr_sl_multiplier}x ATR), TP: {take_profit_price:.{price_precision_digits}f} ({atr_tp_multiplier}x ATR)")
                else: # Fixed percentage
                    stop_loss_price = current_price * (1 + stop_loss_percentage)
                    take_profit_price = current_price * (1 - take_profit_percentage)
                    log_info(f"Calculated Fixed % SL: {stop_loss_price:.{price_precision_digits}f} ({stop_loss_percentage*100:.1f}%), TP: {take_profit_price:.{price_precision_digits}f} ({take_profit_percentage*100:.1f}%)")

                # Adjust SL based on Bearish OB high if it provides tighter stop
                if bearish_ob and isinstance(bearish_ob.get('high'), (float, int)) and bearish_ob['high'] < stop_loss_price:
                     adjusted_sl = bearish_ob['high'] * (1 + 0.0005) # Small buffer above high
                     if adjusted_sl > current_price and adjusted_sl > bearish_ob['high']:
                         stop_loss_price = float(exchange.price_to_precision(symbol, adjusted_sl))
                         log_info(f"Adjusted SL tighter based on Bearish OB high: {stop_loss_price:.{price_precision_digits}f}")
                     else:
                         log_warning(f"Could not adjust SL based on OB high {bearish_ob['high']:.{price_precision_digits}f} as it's too close or invalid.")

                # Final SL/TP validation
                if stop_loss_price <= current_price or take_profit_price >= current_price:
                     log_error(f"Invalid SL/TP calculated: SL {stop_loss_price:.{price_precision_digits}f}, TP {take_profit_price:.{price_precision_digits}f} relative to Price {current_price:.{price_precision_digits}f}. Skipping SHORT entry.")
                     continue

                # Calculate position size
                quantity = calculate_position_size(exchange, symbol, current_price, stop_loss_price, risk_percentage)
                if quantity is None or quantity <= 0:
                    log_error("Failed to calculate valid position size. Skipping SHORT entry.")
                else:
                    # Place entry market order
                    entry_order_result = place_market_order(exchange, symbol, 'sell', quantity, reduce_only=False)
                    is_filled = entry_order_result and (entry_order_result.get('status') == 'closed' or (entry_order_result.get('status') == 'open' and entry_order_result.get('filled', 0) > 0))

                    if is_filled:
                        entry_price_actual = entry_order_result.get('average', current_price)
                        filled_quantity = entry_order_result.get('filled', quantity)

                        log_info(f"Short position entry order filled/processed: ID {entry_order_result.get('id', 'N/A')} at ~{entry_price_actual:.{price_precision_digits}f} for {filled_quantity:.{amount_precision_digits}f}")

                        # Place SL/TP orders
                        sl_order, tp_order = place_sl_tp_orders(exchange, symbol, 'short', filled_quantity, stop_loss_price, take_profit_price)

                        # Update position state
                        position.update({
                            'status': 'short',
                            'entry_price': entry_price_actual,
                            'quantity': filled_quantity,
                            'order_id': entry_order_result.get('id'),
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'entry_time': pd.Timestamp.now(tz='UTC'),
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'tp_order_id': tp_order.get('id') if tp_order else None,
                            # Initialize TSL fields
                            'highest_price_since_entry': None,
                            'lowest_price_since_entry': entry_price_actual,
                            'current_trailing_sl_price': None
                        })
                        save_position_state()
                        display_position_status(position, price_precision_digits, amount_precision_digits)

                        if not (sl_order and sl_order.get('id')) or not (tp_order and tp_order.get('id')):
                            log_warning("Entry successful, but SL and/or TP order placement failed or did not return ID. Monitor position closely!")
                    else:
                        log_error(f"Failed to place or confirm fill for short entry order. Result: {entry_order_result}")

            else: # No entry condition met
                log_info("Conditions not met for new entry.")


        # 7. Wait for the next cycle
        log_info(f"Cycle complete. Waiting for {sleep_interval_seconds} seconds...")
        neon_sleep_timer(sleep_interval_seconds)

    # --- Graceful Shutdown Handling ---
    except KeyboardInterrupt:
        log_info("Keyboard interrupt detected (Ctrl+C). Stopping the bot...")
        save_position_state()  # Save final state before exiting

        # Attempt to cancel open orders for the symbol
        if not SIMULATION_MODE:
            log_info(f"Attempting to cancel all open orders for {symbol}...")
            try:
                # Fetch open orders first (use retry)
                @api_retry_decorator
                def fetch_open_orders_on_exit(exch, sym):
                    return exch.fetch_open_orders(sym)

                open_orders = fetch_open_orders_on_exit(exchange, symbol)

                if not open_orders:
                    log_info("No open orders found to cancel.")
                else:
                    log_warning(f"Found {len(open_orders)} open orders for {symbol}. Attempting cancellation...")
                    cancelled_count = 0
                    failed_count = 0
                    for order in open_orders:
                        order_id = order.get('id')
                        if not order_id: continue # Skip if no ID
                        try:
                            log_info(f"Cancelling order ID: {order_id} (Type: {order.get('type', 'N/A')}, Side: {order.get('side','N/A')})...")
                            # Use cancel helper with retry
                            cancel_order_with_retry(exchange, order_id, symbol)
                            cancelled_count += 1
                            time.sleep(0.3) # Small delay between cancellations
                        except ccxt.OrderNotFound:
                             log_info(f"Order {order_id} already closed/cancelled.")
                             cancelled_count += 1 # Count as effectively cancelled
                        except Exception as cancel_e:
                            # Error logged by helper
                            log_error(f"Failed to cancel order {order_id} on exit after retries.", exc_info=False)
                            failed_count +=1
                    log_info(f"Order cancellation attempt complete. Success/Already Closed: {cancelled_count}, Failed: {failed_count}")
            except Exception as e:
                log_error(f"Error fetching or cancelling open orders on exit: {e}", exc_info=True)
        else:
             log_info("Simulation mode: Skipping order cancellation on exit.")

        break # Exit the main while loop

    # --- Robust Error Handling for the Main Loop ---
    except ccxt.RateLimitExceeded as e:
        log_error(f"Main loop Rate Limit Error: {e}. Waiting longer (default + 60s)...", exc_info=False)
        neon_sleep_timer(sleep_interval_seconds + 60) # Wait longer
    except ccxt.NetworkError as e:
        log_error(f"Main loop Network Error: {e}. Default sleep + retry...", exc_info=False)
        neon_sleep_timer(sleep_interval_seconds) # Rely on retry decorator for backoff
    except ccxt.ExchangeError as e: # Catch other exchange-specific issues
        log_error(f"Main loop Exchange Error: {e}. Default sleep + retry...", exc_info=False)
        # Could add specific handling for maintenance errors etc. here
        neon_sleep_timer(sleep_interval_seconds)
    except Exception as e:
        log_error(f"CRITICAL unexpected error in main loop: {e}", exc_info=True)
        log_info("Attempting to recover by saving state and waiting 60s before next cycle...")
        try:
            save_position_state() # Save state on critical error
        except Exception as save_e:
            log_error(f"Failed to save state during critical error handling: {save_e}", exc_info=True)
        neon_sleep_timer(60) # Wait before trying next cycle

# --- Bot Exit ---
print_shutdown_message()
sys.exit(0)
```
