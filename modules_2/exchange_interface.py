# File: exchange_interface.py
import ccxt
import pandas as pd
import time
import logging
import math # Added for math.isfinite
from decimal import Decimal, getcontext, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP, ROUND_CEILING
from typing import Dict, Any, Optional, Tuple, List, Union

# Assuming app_config.py is in the same directory for these constants
from app_config import CALCULATION_PRECISION, DECIMAL_DISPLAY_PRECISION
from trading_enums import OrderSide, PositionSide # Assuming trading_enums.py is in the same directory

class BybitV5Wrapper:
    """
    Wraps CCXT exchange interactions, focusing on Bybit V5 specifics,
    error handling, Decimal usage, and rate limiting.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.exchange_id = config['exchange'].get('id', 'bybit')
        self.category = config['trading_settings'].get('category', 'linear')
        self.hedge_mode = config['trading_settings'].get('hedge_mode', False)
        self.max_retries = config['exchange'].get('max_retries', 3)
        self.retry_delay = config['exchange'].get('retry_delay_seconds', 5)

        if self.exchange_id != 'bybit':
             self.logger.warning(f"This wrapper is optimized for Bybit V5, but exchange ID is set to '{self.exchange_id}'.")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
        except AttributeError:
             self.logger.critical(f"CCXT exchange class not found for ID: '{self.exchange_id}'. Exiting.")
             raise

        ccxt_config = {
            'apiKey': config['api_credentials']['api_key'],
            'secret': config['api_credentials']['api_secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap' if self.category in ['linear', 'inverse'] else 'spot',
                'adjustForTimeDifference': True,
            },
        }
        if config['exchange'].get('broker_id'):
            ccxt_config['options']['broker_id'] = config['exchange']['broker_id']
        if config['exchange'].get('sandbox_mode', False):
            self.logger.warning("Sandbox mode enabled. Using testnet URLs.")
            ccxt_config['sandboxMode'] = True

        self.exchange = exchange_class(ccxt_config)

        if config['exchange'].get('sandbox_mode', False):
            if hasattr(self.exchange, 'set_sandbox_mode') and not self.exchange.sandbox:
                try: self.exchange.set_sandbox_mode(True)
                except Exception as e: self.logger.error(f"Error calling set_sandbox_mode: {e}")
        
        self.load_markets_with_retry()

    def load_markets_with_retry(self, reload=False):
        retries = 0
        while retries <= self.max_retries:
            try:
                self.exchange.load_markets(reload=reload)
                self.logger.info(f"Markets loaded successfully for {self.exchange_id}.")
                return True
            except ccxt.AuthenticationError:
                self.logger.exception("Authentication failed loading markets. Check API keys.")
                raise
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.OnMaintenance) as e:
                self.logger.warning(f"Network/Availability error loading markets: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeError as e:
                self.logger.exception(f"Exchange error loading markets: {e}. Might retry.")
                if retries >= self.max_retries:
                     self.logger.error("Exchange error loading markets after retries.")
                     raise
            except Exception as e:
                 self.logger.exception(f"Unexpected error loading markets: {e}")
                 raise
            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.retry_delay)
        self.logger.critical(f"Failed to load markets for {self.exchange_id} after {self.max_retries + 1} attempts.")
        raise ccxt.ExchangeError("Failed to load markets after multiple retries.")

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.exchange.markets or symbol not in self.exchange.markets:
                self.logger.warning(f"Market data for {symbol} not loaded. Attempting to load/reload markets.")
                self.load_markets_with_retry(reload=True)
            
            market = self.exchange.market(symbol)
            if not market:
                 self.logger.error(f"Market data for symbol '{symbol}' is null after fetching.")
                 return None

            precision = market.get('precision', {})
            if not precision or 'amount' not in precision or 'price' not in precision:
                 self.logger.warning(f"Precision info incomplete for market {symbol}. Reloading markets.")
                 self.load_markets_with_retry(reload=True)
                 market = self.exchange.market(symbol)
                 precision = market.get('precision', {})
                 if not precision or 'amount' not in precision or 'price' not in precision:
                      self.logger.error(f"Failed to load complete precision info for {symbol} after reload.")
                      return None
            
            if market.get('contract') and 'contractSize' in market:
                 try:
                      cs = market['contractSize']
                      market['contractSize'] = Decimal(str(cs)) if cs is not None else Decimal('NaN')
                 except (InvalidOperation, TypeError):
                      self.logger.error(f"Failed to parse contractSize '{market['contractSize']}' as Decimal for {symbol}.")
                      return None
            return market
        except ccxt.BadSymbol:
            self.logger.error(f"Symbol '{symbol}' not found on {self.exchange_id}.")
            return None
        except ccxt.ExchangeError as e:
             self.logger.error(f"Exchange error fetching market data for {symbol}: {e}", exc_info=True)
             return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching market data for {symbol}: {e}", exc_info=True)
            return None

    def safe_ccxt_call(self, method_name: str, *args, **kwargs) -> Optional[Any]:
        retries = 0
        while retries <= self.max_retries:
            try:
                method = getattr(self.exchange, method_name)
                call_params = kwargs.get('params', {}).copy()
                if self.exchange_id == 'bybit' and self.category in ['linear', 'inverse'] and 'category' not in call_params:
                     methods_requiring_category = [
                         'create_order', 'edit_order', 'cancel_order', 'cancel_all_orders',
                         'fetch_order', 'fetch_open_orders', 'fetch_closed_orders', 'fetch_my_trades',
                         'fetch_position', 'fetch_positions', 'fetch_balance', 'set_leverage', 
                         'set_margin_mode', 'fetch_leverage_tiers', 'fetch_funding_rate', 
                         'fetch_funding_rates', 'fetch_funding_rate_history',
                         'private_post_position_trading_stop', 
                         'private_post_position_switch_margin_mode',
                         'private_post_position_switch_mode',
                     ]
                     if method_name in methods_requiring_category or method_name.startswith('private_post_position'):
                         call_params['category'] = self.category
                
                current_kwargs = kwargs.copy()
                current_kwargs['params'] = call_params
                self.logger.debug(f"Calling CCXT method: {method_name}, Args: {args}, Kwargs: {current_kwargs}")
                result = method(*args, **current_kwargs)
                self.logger.debug(f"CCXT call {method_name} successful. Result snippet: {str(result)[:200]}...")
                return result

            except ccxt.AuthenticationError as e:
                self.logger.error(f"Authentication Error calling {method_name}: {e}. Non-retryable.")
                return None
            except ccxt.PermissionDenied as e:
                 self.logger.error(f"Permission Denied calling {method_name}: {e}. Non-retryable.")
                 return None
            except ccxt.AccountSuspended as e:
                 self.logger.error(f"Account Suspended calling {method_name}: {e}. Non-retryable.")
                 return None
            except ccxt.InvalidOrder as e:
                self.logger.error(f"Invalid Order parameters/state calling {method_name}: {e}. Non-retryable.")
                self.logger.debug(f"Failed Call Details - Method: {method_name}, Args: {args}, Kwargs: {kwargs}")
                return None
            except ccxt.InsufficientFunds as e:
                self.logger.error(f"Insufficient Funds calling {method_name}: {e}. Non-retryable.")
                return None
            except ccxt.BadSymbol as e:
                self.logger.error(f"Invalid Symbol calling {method_name}: {e}. Non-retryable.")
                return None
            except ccxt.BadRequest as e:
                 self.logger.error(f"Bad Request calling {method_name}: {e}. Assuming non-retryable.")
                 self.logger.debug(f"Failed Call Details - Method: {method_name}, Args: {args}, Kwargs: {kwargs}")
                 if self.exchange_id == 'bybit':
                      msg = str(e).lower()
                      non_retryable_codes = ['110007', '110045', '110043', '110014']
                      if any(f"ret_code={code}" in msg or f"retcode={code}" in msg or f"'{code}'" in msg for code in non_retryable_codes):
                           self.logger.error(f"Detected specific non-retryable Bybit error code in message: {msg}")
                 return None
            except ccxt.MarginModeAlreadySet as e:
                 self.logger.warning(f"Margin mode already set as requested: {e}. Considered success for {method_name}.")
                 return {}
            except ccxt.OperationFailed as e:
                 self.logger.error(f"Operation Failed calling {method_name}: {e}. Assuming non-retryable.")
                 return None
            except ccxt.RateLimitExceeded as e:
                self.logger.warning(f"Rate Limit Exceeded calling {method_name}: {e}. Retrying... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.NetworkError as e:
                self.logger.warning(f"Network Error calling {method_name}: {e}. Retrying... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeNotAvailable as e:
                self.logger.warning(f"Exchange Not Available calling {method_name}: {e}. Retrying... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.OnMaintenance as e:
                 self.logger.warning(f"Exchange On Maintenance calling {method_name}: {e}. Retrying... (Attempt {retries + 1}/{self.max_retries + 1})")
            except ccxt.ExchangeError as e:
                msg = str(e).lower()
                non_retryable_msgs = ['position idx not match', 'insufficient available balance', 'risk limit', 'order cost not available', 'cannot be modified', 'order status is incorrect']
                retryable_codes = ['10002', '10006', '10016']
                if any(term in msg for term in non_retryable_msgs) and not any(f"ret_code={code}" in msg or f"retcode={code}" in msg for code in retryable_codes):
                     self.logger.error(f"Potentially non-retryable Exchange Error calling {method_name}: {e}.")
                     return None
                elif any(f"ret_code={code}" in msg or f"retcode={code}" in msg for code in retryable_codes):
                     self.logger.warning(f"Retryable Exchange Error code detected calling {method_name}: {e}. Retrying...")
                else:
                     self.logger.warning(f"Generic Exchange Error calling {method_name}: {e}. Retrying... (Attempt {retries + 1}/{self.max_retries + 1})")
            except Exception as e:
                 self.logger.error(f"Unexpected error during CCXT call '{method_name}': {e}", exc_info=True)
                 return None

            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.retry_delay)
            else:
                self.logger.error(f"CCXT call '{method_name}' failed after {self.max_retries + 1} attempts.")
                return None
        return None # Should ideally not be reached

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        self.logger.info(f"Fetching {limit} OHLCV candles for {symbol} ({timeframe})...")
        ohlcv = self.safe_ccxt_call('fetch_ohlcv', symbol, timeframe, limit=limit)
        if ohlcv is None or not ohlcv:
            return None
        try:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                 self.logger.warning(f"Fetched OHLCV data for {symbol} resulted in an empty DataFrame.")
                 return df
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            with getcontext() as ctx: # Ensure high precision for conversion
                ctx.prec = CALCULATION_PRECISION
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal('NaN'))
            
            nan_mask = df[['open', 'high', 'low', 'close']].isnull().any(axis=1) | df[['open', 'high', 'low', 'close']].applymap(lambda x: x.is_nan()).any(axis=1)
            if nan_mask.any():
                 self.logger.warning(f"{nan_mask.sum()} rows with NaN values found in fetched OHLCV data for {symbol}.")
            zero_price_mask = (df[['open', 'high', 'low', 'close']] <= Decimal(0)).any(axis=1)
            if zero_price_mask.any():
                 self.logger.warning(f"{zero_price_mask.sum()} rows with zero or negative values found in OHLCV prices for {symbol}.")
            
            self.logger.info(f"Successfully fetched and processed {len(df)} OHLCV candles for {symbol}.")
            return df
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
            return None

    def fetch_balance(self) -> Optional[Dict[str, Decimal]]:
        self.logger.debug("Fetching account balance...")
        balance_data = self.safe_ccxt_call('fetch_balance')
        if balance_data is None: return None
        balances = {}
        try:
            if self.exchange_id == 'bybit' and self.category in ['linear', 'inverse', 'spot']:
                account_list = balance_data.get('info', {}).get('result', {}).get('list', [])
                if not account_list: self.logger.warning("Balance response structure unexpected (no 'list').")
                else:
                    account_type_map = {'linear': 'CONTRACT', 'inverse': 'CONTRACT', 'spot': 'SPOT'}
                    target_account_type = account_type_map.get(self.category)
                    unified_account, specific_account = None, None
                    for acc in account_list:
                        acc_type = acc.get('accountType')
                        if acc_type == 'UNIFIED': unified_account = acc; break
                        elif acc_type == target_account_type: specific_account = acc
                    account_to_parse = unified_account or specific_account
                    if account_to_parse:
                        self.logger.debug(f"Parsing balance from account type: {account_to_parse.get('accountType')}")
                        coin_data = account_to_parse.get('coin', [])
                        with getcontext() as ctx:
                            ctx.prec = CALCULATION_PRECISION
                            for coin_info in coin_data:
                                asset, equity_str = coin_info.get('coin'), coin_info.get('equity')
                                if asset and equity_str is not None and equity_str != '':
                                    try: balances[asset] = Decimal(str(equity_str))
                                    except InvalidOperation: self.logger.warning(f"Could not convert balance for {asset} to Decimal: {equity_str}")
                    else:
                         self.logger.warning(f"Could not find relevant account type ('UNIFIED' or '{target_account_type}') in Bybit V5 balance response.")
            if not balances:
                self.logger.debug("Using standard CCXT 'total' balance parsing.")
                with getcontext() as ctx:
                    ctx.prec = CALCULATION_PRECISION
                    for asset, bal_info in balance_data.get('total', {}).items():
                        if bal_info is not None:
                            try: balances[asset] = Decimal(str(bal_info))
                            except InvalidOperation: self.logger.warning(f"Could not convert balance (fallback) for {asset} to Decimal: {bal_info}")
            if not balances:
                 self.logger.warning("Parsed balance data is empty."); self.logger.debug(f"Raw balance data: {balance_data}"); return {}
            else:
                 self.logger.info(f"Balance fetched. Assets: {list(balances.keys())}")
                 quote_asset = self.config['trading_settings']['quote_asset']
                 if quote_asset in balances: self.logger.info(f"{quote_asset} Equity: {balances[quote_asset]:.{DECIMAL_DISPLAY_PRECISION}f}")
                 else: self.logger.warning(f"Configured quote asset '{quote_asset}' not found in balances.")
            return balances
        except Exception as e:
            self.logger.error(f"Error parsing balance data: {e}", exc_info=True); self.logger.debug(f"Raw balance data: {balance_data}"); return None

    def fetch_positions(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        target_symbol = symbol or self.config['trading_settings']['symbol']
        self.logger.debug(f"Fetching positions for symbol: {target_symbol}...")
        params, symbols_arg = {}, [target_symbol]
        if self.exchange_id == 'bybit':
            params['category'], params['symbol'], symbols_arg = self.category, target_symbol, None
        positions_data = self.safe_ccxt_call('fetch_positions', symbols=symbols_arg, params=params)
        if positions_data is None: return None
        processed_positions = []
        try:
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION
                for pos in positions_data:
                    if pos.get('symbol') != target_symbol: continue
                    size_str, contracts_val = pos.get('info', {}).get('size'), pos.get('contracts')
                    size_dec, is_active = Decimal('0'), False
                    if size_str is not None and size_str != '':
                        try: size_dec = Decimal(str(size_str)); is_active = size_dec.is_finite() and size_dec != Decimal(0)
                        except InvalidOperation: self.logger.warning(f"Could not parse position size '{size_str}' from info as Decimal for {target_symbol}.")
                    elif contracts_val is not None:
                         try: size_dec = Decimal(str(contracts_val)); is_active = size_dec.is_finite() and size_dec != Decimal(0)
                         except InvalidOperation: self.logger.warning(f"Could not parse position contracts '{contracts_val}' as Decimal for {target_symbol}.")
                    if not is_active: self.logger.debug(f"Skipping zero-size position for: {target_symbol}"); continue
                    
                    processed = pos.copy(); processed['contracts'] = size_dec
                    decimal_fields_std = ['contractSize', 'entryPrice', 'leverage', 'liquidationPrice', 'markPrice', 'notional', 'unrealizedPnl', 'initialMargin', 'maintenanceMargin', 'initialMarginPercentage', 'maintenanceMarginPercentage', 'marginRatio', 'collateral']
                    decimal_fields_info = ['avgPrice', 'cumRealisedPnl', 'liqPrice', 'markPrice', 'positionValue', 'stopLoss', 'takeProfit', 'trailingStop', 'unrealisedPnl', 'positionIM', 'positionMM', 'createdTime', 'updatedTime']
                    for field in decimal_fields_std:
                        if field in processed and processed[field] is not None:
                            try: processed[field] = Decimal(str(processed[field]))
                            except InvalidOperation: self.logger.warning(f"Could not convert std pos field '{field}' ({processed[field]}) to Decimal."); processed[field] = Decimal('NaN')
                    if 'info' in processed and isinstance(processed['info'], dict):
                        info = processed['info']
                        for field in decimal_fields_info:
                            if field in info and info[field] is not None and info[field] != '':
                                try: info[field] = Decimal(str(info[field]))
                                except InvalidOperation: self.logger.warning(f"Could not convert pos info field '{field}' ({info[field]}) to Decimal."); info[field] = Decimal('NaN')
                    
                    side_enum = PositionSide.NONE
                    if 'info' in processed and isinstance(processed['info'], dict) and 'side' in processed['info']:
                        side_str = str(processed['info']['side']).lower()
                        if side_str == 'buy': side_enum = PositionSide.LONG
                        elif side_str == 'sell': side_enum = PositionSide.SHORT
                        elif side_str == 'none': side_enum = PositionSide.NONE
                    if side_enum == PositionSide.NONE and 'side' in processed and processed['side']:
                         side_str_std = str(processed['side']).lower()
                         if side_str_std == 'long': side_enum = PositionSide.LONG
                         elif side_str_std == 'short': side_enum = PositionSide.SHORT
                    if side_enum == PositionSide.NONE and size_dec != Decimal(0): self.logger.warning(f"Position for {target_symbol} has size {size_dec} but side is 'None'.")
                    processed['side'] = side_enum.value

                    pos_idx_val = processed.get('info', {}).get('positionIdx')
                    if pos_idx_val is not None:
                        try: processed['positionIdx'] = int(pos_idx_val)
                        except (ValueError, TypeError): self.logger.warning(f"Could not parse positionIdx '{pos_idx_val}'."); processed['positionIdx'] = None
                    else: processed['positionIdx'] = 0 if not self.hedge_mode else None
                    processed_positions.append(processed)
            self.logger.info(f"Fetched {len(processed_positions)} active position(s) for {target_symbol}.")
            return processed_positions
        except Exception as e:
            self.logger.error(f"Error processing position data for {target_symbol}: {e}", exc_info=True); self.logger.debug(f"Raw positions data: {positions_data}"); return None

    def format_value_for_api(self, symbol: str, value_type: str, value: Decimal,
                              rounding_mode: str = ROUND_HALF_UP) -> str:
        if not isinstance(value, Decimal): raise ValueError(f"Invalid input value type: {type(value)}. Expected Decimal.")
        if not value.is_finite(): raise ValueError(f"Invalid Decimal value for formatting: {value}. Must be finite.")
        market = self.get_market(symbol)
        if not market: raise ValueError(f"Market data not found for {symbol}, cannot format value.")
        value_float = float(value)
        try:
            if value_type == 'amount': formatted_value = self.exchange.amount_to_precision(symbol, value_float)
            elif value_type == 'price': formatted_value = self.exchange.price_to_precision(symbol, value_float)
            else: raise ValueError(f"Invalid value_type: {value_type}. Use 'amount' or 'price'.")
            return formatted_value
        except ccxt.ExchangeError as e: self.logger.error(f"CCXT error formatting {value_type} ('{value}') for {symbol}: {e}"); raise ValueError(f"CCXT error formatting {value_type}") from e
        except Exception as e: self.logger.error(f"Unexpected error formatting {value_type} ('{value}') for {symbol}: {e}", exc_info=True); raise ValueError(f"Unexpected error formatting {value_type}") from e

    def quantize_value(self, value: Decimal, precision_type: str, market: Dict[str, Any],
                        rounding_mode: str = ROUND_HALF_UP) -> Optional[Decimal]:
        if not isinstance(value, Decimal) or not value.is_finite(): self.logger.error(f"Invalid value for quantization: {value}"); return None
        if not market or 'precision' not in market: self.logger.error("Market data or precision missing for quantization."); return None
        precision_val_str = market['precision'].get(precision_type)
        if precision_val_str is None: self.logger.error(f"Precision value for type '{precision_type}' not found."); return None
        try:
            tick_size = Decimal(str(precision_val_str))
            if not tick_size.is_finite() or tick_size <= 0: raise InvalidOperation("Invalid tick/step size")
            with getcontext() as ctx:
                ctx.prec = CALCULATION_PRECISION
                quantized_value = (value / tick_size).quantize(Decimal('1'), rounding=rounding_mode) * tick_size
            return quantized_value
        except (InvalidOperation, ValueError, TypeError) as e: self.logger.error(f"Error quantizing value {value} with precision '{precision_val_str}': {e}"); return None

    def create_order(self, symbol: str, order_type: str, side: OrderSide, amount: Decimal,
                     price: Optional[Decimal] = None, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        self.logger.info(f"Attempting to create {side.value} {order_type} order for {amount} {symbol} @ {price or 'market'}...")
        if amount <= 0: self.logger.error(f"Amount must be positive ({amount})."); return None
        if order_type.lower() not in ['market', 'stop_market', 'take_profit_market'] and (price is None or price <= 0):
             self.logger.error(f"Price must be positive for non-market order ({price})."); return None
        market = self.get_market(symbol)
        if not market: self.logger.error(f"Market data for {symbol} not found."); return None
        try:
            amount_str = self.format_value_for_api(symbol, 'amount', amount, rounding_mode=ROUND_DOWN)
            if Decimal(amount_str) <= 0: self.logger.error(f"Order amount {amount} after formatting became non-positive ({amount_str})."); return None
            price_str = None
            if price is not None and order_type.lower() not in ['market', 'stop_market', 'take_profit_market']:
                price_str = self.format_value_for_api(symbol, 'price', price, rounding_mode=ROUND_HALF_UP)
            self.logger.debug(f"Formatted order values: Amount='{amount_str}', Price='{price_str}'")
        except ValueError as e: self.logger.error(f"Error formatting order values: {e}"); return None
        except Exception as e: self.logger.error(f"Unexpected error during value formatting: {e}", exc_info=True); return None
        
        order_params = params.copy() if params else {}
        if self.exchange_id == 'bybit':
            if self.hedge_mode:
                if 'positionIdx' not in order_params:
                    if not order_params.get('reduceOnly', False):
                         order_params['positionIdx'] = 1 if side == OrderSide.BUY else 2
            else: order_params.setdefault('positionIdx', 0)
        
        call_args = [symbol, order_type, side.value, amount_str]
        if order_type.lower() in ['market', 'stop_market', 'take_profit_market']: call_args.append(None)
        else:
             if price_str is None: self.logger.error(f"Price required for '{order_type}' but not provided/formatted."); return None
             call_args.append(price_str)
        
        order_result = self.safe_ccxt_call('create_order', *call_args, params=order_params)
        if order_result: self.logger.info(f"Order creation request successful. ID: {order_result.get('id')}, Status: {order_result.get('status', 'unknown')}")
        else: self.logger.error(f"Failed to create {side.value} {order_type} order for {symbol}.")
        return order_result

    def set_leverage(self, symbol: str, leverage: Decimal) -> bool:
        self.logger.info(f"Setting leverage for {symbol} to {leverage}x...")
        if not isinstance(leverage, Decimal) or not leverage.is_finite() or leverage <= 0:
             self.logger.error(f"Invalid leverage value: {leverage}."); return False
        leverage_float, leverage_str = float(leverage), f"{int(leverage)}"
        params = {}
        if self.exchange_id == 'bybit':
            params['buyLeverage'], params['sellLeverage'] = leverage_str, leverage_str
        result = self.safe_ccxt_call('set_leverage', leverage_float, symbol, params=params)
        if result is not None:
             self.logger.info(f"Leverage for {symbol} set to {leverage}x request sent.")
             if self.exchange_id == 'bybit' and isinstance(result, dict) and 'info' in result:
                  ret_code, ret_msg = result['info'].get('retCode'), result['info'].get('retMsg', 'Unknown')
                  if ret_code == 0: self.logger.info("Bybit API confirmed successful leverage setting."); return True
                  else: self.logger.warning(f"Bybit leverage response - Code: {ret_code}, Msg: {ret_msg}. Assuming success if no modification needed."); return True
             return True
        return False

    def set_protection(self, symbol: str, stop_loss: Optional[Decimal] = None,
                       take_profit: Optional[Decimal] = None, trailing_stop: Optional[Dict[str, Decimal]] = None,
                       position_idx: Optional[int] = None) -> bool:
        action_parts = []
        if stop_loss is not None: action_parts.append(f"SL={stop_loss}")
        if take_profit is not None: action_parts.append(f"TP={take_profit}")
        if trailing_stop is not None: action_parts.append(f"TSL={trailing_stop}")
        if not action_parts: self.logger.warning("set_protection called with no levels."); return False
        self.logger.info(f"Attempting to set protection for {symbol} (Idx: {position_idx if position_idx is not None else 'N/A'}): {' / '.join(action_parts)}")
        market = self.get_market(symbol)
        if not market: self.logger.error(f"Cannot set protection: Market data for {symbol} not found."); return False
        
        params = {'symbol': symbol}
        try:
            if stop_loss is not None:
                if stop_loss <= 0: raise ValueError("Stop loss must be positive.")
                quantized_sl = self.quantize_value(stop_loss, 'price', market, ROUND_HALF_UP)
                if quantized_sl is None: raise ValueError("Failed to quantize stop loss.")
                params['stopLoss'] = self.format_value_for_api(symbol, 'price', quantized_sl)
            if take_profit is not None:
                if take_profit <= 0: raise ValueError("Take profit must be positive.")
                quantized_tp = self.quantize_value(take_profit, 'price', market, ROUND_HALF_UP)
                if quantized_tp is None: raise ValueError("Failed to quantize take profit.")
                params['takeProfit'] = self.format_value_for_api(symbol, 'price', quantized_tp)
            if trailing_stop:
                ts_value = trailing_stop.get('distance') or trailing_stop.get('value')
                ts_active_price = trailing_stop.get('activation_price')
                if ts_value is not None:
                     if not isinstance(ts_value, Decimal) or not ts_value.is_finite() or ts_value <= 0: raise ValueError("TSL distance must be positive finite Decimal.")
                     quantized_ts_dist = self.quantize_value(ts_value, 'price', market, ROUND_HALF_UP)
                     if quantized_ts_dist is None: raise ValueError("Failed to quantize TSL distance.")
                     params['trailingStop'] = self.format_value_for_api(symbol, 'price', quantized_ts_dist)
                if ts_active_price is not None:
                     if not isinstance(ts_active_price, Decimal) or not ts_active_price.is_finite() or ts_active_price <= 0: raise ValueError("TSL activation price must be positive finite Decimal.")
                     quantized_ts_active = self.quantize_value(ts_active_price, 'price', market, ROUND_HALF_UP)
                     if quantized_ts_active is None: raise ValueError("Failed to quantize TSL activation price.")
                     params['activePrice'] = self.format_value_for_api(symbol, 'price', quantized_ts_active)
        except ValueError as e: self.logger.error(f"Invalid protection parameter: {e}"); return False
        except Exception as e: self.logger.error(f"Error formatting protection params: {e}", exc_info=True); return False

        if self.hedge_mode:
            if position_idx is None: self.logger.error("Hedge mode: positionIdx required for set_protection."); return False
            params['positionIdx'] = position_idx
        else: params.setdefault('positionIdx', 0)
        
        if not hasattr(self.exchange, 'private_post_position_trading_stop'):
             self.logger.error("CCXT missing 'private_post_position_trading_stop'. Cannot set protection."); return False
        result = self.safe_ccxt_call('private_post_position_trading_stop', params=params)
        if result is not None:
            ret_code, ret_msg = result.get('retCode'), result.get('retMsg', 'No message')
            if ret_code == 0: self.logger.info(f"Protection levels set/updated successfully for {symbol}."); return True
            else: self.logger.error(f"Failed to set protection for {symbol}. Code: {ret_code}, Msg: {ret_msg}, Extra: {result.get('retExtInfo', {})}, Params: {params}"); return False
        else: self.logger.error(f"API call failed for setting protection on {symbol}."); return False

```

```python
