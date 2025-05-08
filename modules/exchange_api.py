# File: exchange_api.py
import ccxt
import requests
import time
import logging
import os
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Tuple, List, Union

# Import constants and utilities
import constants
import utils

# Fetch API keys from environment where needed
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with error handling."""
    lg = logger
    quote_currency = config.get('quote_currency', 'USDT') # Needed for balance logging
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'fetchTickerTimeout': 10000,
                'fetchBalanceTimeout': 15000,
                'createOrderTimeout': 20000,
                'cancelOrderTimeout': 15000,
                'fetchPositionsTimeout': 15000,
                'fetchOHLCVTimeout': 15000,
            }
        }

        exchange_id = config.get("exchange_id", "bybit").lower()
        if not hasattr(ccxt, exchange_id):
             lg.error(f"Exchange ID '{exchange_id}' not found in CCXT.")
             return None
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        if config.get('use_sandbox'):
            lg.warning(f"{constants.NEON_YELLOW}USING SANDBOX MODE (Testnet){constants.RESET}")
            if hasattr(exchange, 'set_sandbox_mode'):
                try:
                    exchange.set_sandbox_mode(True)
                except Exception as sandbox_err:
                     lg.warning(f"Error calling set_sandbox_mode(True) for {exchange.id}: {sandbox_err}. Attempting manual URL override.")
                     if exchange.id == 'bybit':
                          lg.info("Attempting manual Bybit testnet URL override...")
                          exchange.urls['api'] = 'https://api-testnet.bybit.com'
            else:
                lg.warning(f"{exchange.id} does not support set_sandbox_mode via ccxt. Ensure API keys are for Testnet.")
                if exchange.id == 'bybit':
                     lg.info("Attempting manual Bybit testnet URL override...")
                     exchange.urls['api'] = 'https://api-testnet.bybit.com'

        lg.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox')}")

        # Test connection by fetching balance
        account_type_to_test = 'CONTRACT' # For Bybit V5
        lg.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            params = {}
            if exchange.id == 'bybit':
                 params={'type': account_type_to_test}
            balance = exchange.fetch_balance(params=params)
            available_quote = balance.get(quote_currency, {}).get('free', 'N/A')
            lg.info(f"{constants.NEON_GREEN}Successfully connected and fetched initial balance.{constants.RESET} (Example: {quote_currency} available: {available_quote})")
        except ccxt.AuthenticationError as auth_err:
             lg.error(f"{constants.NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{constants.RESET}")
             lg.error(f"{constants.NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on the exchange.{constants.RESET}")
             return None
        except ccxt.ExchangeError as balance_err:
             lg.warning(f"{constants.NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{constants.RESET}")
             try:
                  balance = exchange.fetch_balance()
                  available_quote = balance.get(quote_currency, {}).get('free', 'N/A')
                  lg.info(f"{constants.NEON_GREEN}Successfully fetched balance using default parameters.{constants.RESET} (Example: {quote_currency} available: {available_quote})")
             except Exception as fallback_err:
                  lg.warning(f"{constants.NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Check API permissions/account type.{constants.RESET}")
        except Exception as balance_err:
             lg.warning(f"{constants.NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type/network.{constants.RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        lg.error(f"{constants.NEON_RED}CCXT Authentication Error during initialization: {e}{constants.RESET}")
        lg.error(f"{constants.NEON_RED}>> Check API Key/Secret format and validity in your .env file.{constants.RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{constants.NEON_RED}CCXT Exchange Error initializing: {e}{constants.RESET}")
    except ccxt.NetworkError as e:
        lg.error(f"{constants.NEON_RED}CCXT Network Error initializing: {e}{constants.RESET}")
    except Exception as e:
        lg.error(f"{constants.NEON_RED}Failed to initialize CCXT exchange: {e}{constants.RESET}", exc_info=True)

    return None


def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks."""
    lg = logger
    attempts = 0
    while attempts <= constants.MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            lg.debug(f"Ticker data for {symbol}: {ticker}")

            price = None
            last_price = ticker.get('last')
            bid_price = ticker.get('bid')
            ask_price = ticker.get('ask')

            if last_price is not None:
                try:
                    p = Decimal(str(last_price));
                    if p > 0: price = p; lg.debug(f"Using 'last' price: {p}")
                except (InvalidOperation, ValueError, TypeError): lg.warning(f"Invalid 'last' price format: {last_price}")

            if price is None and bid_price is not None and ask_price is not None:
                try:
                    bid = Decimal(str(bid_price)); ask = Decimal(str(ask_price))
                    if bid > 0 and ask > 0 and ask >= bid: price = (bid + ask) / 2; lg.debug(f"Using bid/ask midpoint: {price}")
                    elif ask > 0: price = ask; lg.debug(f"Using 'ask' price (bid invalid): {price}")
                    elif bid > 0: price = bid; lg.debug(f"Using 'bid' price (ask invalid): {price}")
                except (InvalidOperation, ValueError, TypeError): lg.warning(f"Invalid bid/ask format: {bid_price}, {ask_price}")

            if price is None and ask_price is not None:
                 try: p = Decimal(str(ask_price));
                      if p > 0: price = p; lg.warning(f"Using 'ask' price fallback: {p}")
                 except (InvalidOperation, ValueError, TypeError): lg.warning(f"Invalid 'ask' price format: {ask_price}")

            if price is None and bid_price is not None:
                 try: p = Decimal(str(bid_price));
                      if p > 0: price = p; lg.warning(f"Using 'bid' price fallback: {p}")
                 except (InvalidOperation, ValueError, TypeError): lg.warning(f"Invalid 'bid' price format: {bid_price}")

            if price is not None and price > 0: return price
            else: lg.warning(f"Failed to get a valid price from ticker data on attempt {attempts + 1}.")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            lg.warning(f"{constants.NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{constants.RESET}")
        except ccxt.RateLimitExceeded as e:
            lg.warning(f"{constants.NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting longer...{constants.RESET}")
            time.sleep(constants.RETRY_DELAY_SECONDS * 5)
            attempts += 1; continue
        except ccxt.ExchangeError as e:
            lg.error(f"{constants.NEON_RED}Exchange error fetching price for {symbol}: {e}{constants.RESET}"); return None
        except Exception as e:
            lg.error(f"{constants.NEON_RED}Unexpected error fetching price for {symbol}: {e}{constants.RESET}", exc_info=True); return None

        attempts += 1
        if attempts <= constants.MAX_API_RETRIES: time.sleep(constants.RETRY_DELAY_SECONDS)

    lg.error(f"{constants.NEON_RED}Failed to fetch a valid current price for {symbol} after {constants.MAX_API_RETRIES + 1} attempts.{constants.RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> Optional['pd.DataFrame']:
    """Fetch OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__)
    # Import pandas here as it's only needed for this function in this module
    import pandas as pd
    from datetime import datetime # Import datetime for timestamp conversion

    try:
        if not exchange.has['fetchOHLCV']:
             lg.error(f"Exchange {exchange.id} does not support fetchOHLCV."); return pd.DataFrame()

        ohlcv = None
        for attempt in range(constants.MAX_API_RETRIES + 1):
             try:
                  lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{constants.MAX_API_RETRIES + 1})")
                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

                  if ohlcv is not None and isinstance(ohlcv, list):
                    if not ohlcv: lg.warning(f"fetch_ohlcv returned an empty list for {symbol} (Attempt {attempt+1}). Retrying...")
                    else: break # Success
                  else: lg.warning(f"fetch_ohlcv returned invalid data (type: {type(ohlcv)}) for {symbol} (Attempt {attempt+1}). Retrying...")

             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                  if attempt < constants.MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {constants.RETRY_DELAY_SECONDS}s...")
                      time.sleep(constants.RETRY_DELAY_SECONDS)
                  else: lg.error(f"{constants.NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{constants.RESET}"); raise e
             except ccxt.RateLimitExceeded as e:
                 wait_time = constants.RETRY_DELAY_SECONDS * 5
                 lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                 time.sleep(wait_time)
             except ccxt.ExchangeError as e:
                 lg.error(f"{constants.NEON_RED}Exchange error fetching klines for {symbol}: {e}{constants.RESET}")
                 if "symbol" in str(e).lower(): raise e
                 if attempt < constants.MAX_API_RETRIES: time.sleep(constants.RETRY_DELAY_SECONDS)
                 else: raise e
             except Exception as e:
                lg.error(f"{constants.NEON_RED}Unexpected error during kline fetch attempt {attempt+1} for {symbol}: {e}{constants.RESET}", exc_info=True)
                if attempt < constants.MAX_API_RETRIES: time.sleep(constants.RETRY_DELAY_SECONDS)
                else: raise e

        if not ohlcv or not isinstance(ohlcv, list):
            lg.warning(f"{constants.NEON_YELLOW}No valid kline data returned for {symbol} {timeframe} after retries.{constants.RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"{constants.NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe}.{constants.RESET}"); return df

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0: lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
             lg.warning(f"{constants.NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{constants.RESET}"); return pd.DataFrame()

        df.sort_index(inplace=True)
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e: lg.error(f"{constants.NEON_RED}Network error occurred during kline processing for {symbol}: {e}{constants.RESET}")
    except ccxt.ExchangeError as e: lg.error(f"{constants.NEON_RED}Exchange error occurred during kline processing for {symbol}: {e}{constants.RESET}")
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error processing klines for {symbol}: {e}{constants.RESET}", exc_info=True)
    return pd.DataFrame()


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries and basic validation."""
    lg = logger
    attempts = 0
    while attempts <= constants.MAX_API_RETRIES:
        try:
            if not exchange.has['fetchOrderBook']:
                 lg.error(f"Exchange {exchange.id} does not support fetchOrderBook."); return None

            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts+1}/{constants.MAX_API_RETRIES + 1})")
            orderbook = exchange.fetch_order_book(symbol, limit=limit)

            if not orderbook: lg.warning(f"fetch_order_book returned None/empty for {symbol} (Attempt {attempts+1}).")
            elif not isinstance(orderbook, dict): lg.warning(f"{constants.NEON_YELLOW}Invalid orderbook type received for {symbol}. Expected dict, got {type(orderbook)}. Attempt {attempts + 1}.{constants.RESET}")
            elif 'bids' not in orderbook or 'asks' not in orderbook: lg.warning(f"{constants.NEON_YELLOW}Invalid orderbook structure for {symbol}: missing 'bids' or 'asks'. Attempt {attempts + 1}. Response keys: {list(orderbook.keys())}{constants.RESET}")
            elif not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list): lg.warning(f"{constants.NEON_YELLOW}Invalid orderbook structure for {symbol}: 'bids' or 'asks' are not lists. Attempt {attempts + 1}. bids type: {type(orderbook['bids'])}, asks type: {type(orderbook['asks'])}{constants.RESET}")
            elif not orderbook['bids'] and not orderbook['asks']: lg.warning(f"{constants.NEON_YELLOW}Orderbook received but bids and asks lists are both empty for {symbol}. (Attempt {attempts + 1}).{constants.RESET}"); return orderbook
            else: lg.debug(f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks."); return orderbook

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e: lg.warning(f"{constants.NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying...{constants.RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = constants.RETRY_DELAY_SECONDS * 5
            lg.warning(f"Rate limit exceeded fetching orderbook for {symbol}. Retrying in {wait_time}s..."); time.sleep(wait_time); attempts += 1; continue
        except ccxt.ExchangeError as e:
            lg.error(f"{constants.NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{constants.RESET}")
            if "symbol" in str(e).lower(): return None
        except Exception as e:
            lg.error(f"{constants.NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{constants.RESET}", exc_info=True); return None

        attempts += 1
        if attempts <= constants.MAX_API_RETRIES: time.sleep(constants.RETRY_DELAY_SECONDS)

    lg.error(f"{constants.NEON_RED}Max retries reached fetching orderbook for {symbol}.{constants.RESET}")
    return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency with retries and robust parsing."""
    lg = logger
    for attempt in range(constants.MAX_API_RETRIES + 1):
        try:
            balance_info = None
            account_types_to_try = []
            if exchange.id == 'bybit': account_types_to_try = ['CONTRACT', 'UNIFIED']

            found_structure = False
            for acc_type in account_types_to_try:
                 try:
                     lg.debug(f"Fetching balance using params={{'type': '{acc_type}'}} for {currency}... (Attempt {attempt+1})")
                     balance_info = exchange.fetch_balance(params={'type': acc_type})
                     if currency in balance_info and balance_info[currency].get('free') is not None:
                         found_structure = True; break
                     elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             if isinstance(account.get('coin'), list):
                                 if any(coin_data.get('coin') == currency for coin_data in account['coin']):
                                     found_structure = True; break
                         if found_structure: break
                     lg.debug(f"Currency '{currency}' not directly found using type '{acc_type}'. Checking V5 structure...")
                 except (ccxt.ExchangeError, ccxt.AuthenticationError) as e: lg.debug(f"Error fetching balance for type '{acc_type}': {e}. Trying next.") ; continue
                 except Exception as e: lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}. Trying next."); continue

            if not found_structure:
                 lg.debug(f"Fetching balance using default parameters for {currency}... (Attempt {attempt+1})")
                 try: balance_info = exchange.fetch_balance()
                 except Exception as e: lg.error(f"{constants.NEON_RED}Failed to fetch balance using default parameters: {e}{constants.RESET}"); raise e

            if balance_info:
                available_balance_str = None
                if currency in balance_info and balance_info[currency].get('free') is not None:
                    available_balance_str = str(balance_info[currency]['free']); lg.debug(f"Found balance via standard ['{currency}']['free']: {available_balance_str}")
                elif not available_balance_str and 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                    for account in balance_info['info']['result']['list']:
                        if isinstance(account.get('coin'), list):
                            for coin_data in account['coin']:
                                 if coin_data.get('coin') == currency:
                                     free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                     if free is not None: available_balance_str = str(free); lg.debug(f"Found balance via Bybit V5 nested ['available...']: {available_balance_str}"); break
                            if available_balance_str is not None: break
                    if not available_balance_str: lg.warning(f"{currency} balance details not found within Bybit V5 'info.result.list[].coin[]'.")
                elif not available_balance_str and 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                     available_balance_str = str(balance_info['free'][currency]); lg.debug(f"Found balance via top-level 'free' dict: {available_balance_str}")

                if available_balance_str is None:
                     total_balance = balance_info.get(currency, {}).get('total')
                     if total_balance is not None: lg.warning(f"{constants.NEON_YELLOW}Using 'total' balance ({total_balance}) as fallback for available {currency}.{constants.RESET}"); available_balance_str = str(total_balance)
                     else: lg.error(f"{constants.NEON_RED}Could not determine any balance ('free' or 'total') for {currency}.{constants.RESET}"); lg.debug(f"Full balance_info structure: {balance_info}")

                if available_balance_str is not None:
                    try:
                        final_balance = Decimal(available_balance_str)
                        if final_balance >= 0: lg.info(f"Available {currency} balance: {final_balance:.4f}"); return final_balance
                        else: lg.error(f"Parsed balance for {currency} is negative ({final_balance}).")
                    except (InvalidOperation, ValueError, TypeError) as e: lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
            else: lg.error(f"Balance info was None after fetch attempt {attempt + 1}.")

            raise ccxt.ExchangeError("Balance parsing failed or data missing") # Trigger retry

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.warning(f"Network error fetching balance: {e}. Retrying ({attempt+1}/{constants.MAX_API_RETRIES})...")
        except ccxt.RateLimitExceeded as e:
            wait_time = constants.RETRY_DELAY_SECONDS * 5
            lg.warning(f"Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s ({attempt+1}/{constants.MAX_API_RETRIES})...")
            time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: lg.error(f"{constants.NEON_RED}Authentication error fetching balance: {e}. Aborting balance fetch.{constants.RESET}"); return None
        except ccxt.ExchangeError as e: lg.warning(f"Exchange error fetching balance: {e}. Retrying ({attempt+1}/{constants.MAX_API_RETRIES})...")
        except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error fetching balance: {e}{constants.RESET}", exc_info=True)

        if attempt < constants.MAX_API_RETRIES: time.sleep(constants.RETRY_DELAY_SECONDS)

    lg.error(f"{constants.NEON_RED}Failed to fetch balance for {currency} after {constants.MAX_API_RETRIES + 1} attempts.{constants.RESET}")
    return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information (precision, limits, type) using exchange.market()."""
    lg = logger
    try:
        if not exchange.markets or symbol not in exchange.markets:
             lg.info(f"Market info for {symbol} not loaded or symbol not found, reloading markets...")
             try: exchange.load_markets(reload=True)
             except Exception as load_err: lg.error(f"{constants.NEON_RED}Failed to reload markets: {load_err}{constants.RESET}"); return None

        if symbol not in exchange.markets:
             lg.error(f"{constants.NEON_RED}Market {symbol} still not found after reloading.{constants.RESET}")
             if '/' in symbol:
                 base, quote = symbol.split('/', 1)
                 perp_sym_usdt = f"{symbol}:USDT"; perp_sym_perp = f"{base}-PERP"
                 possible_matches = [s for s in exchange.markets if s.startswith(base) and ('PERP' in s or ':USDT' in s)]
                 if perp_sym_usdt in exchange.markets: lg.warning(f"{constants.NEON_YELLOW}Did you mean '{perp_sym_usdt}'?{constants.RESET}")
                 elif perp_sym_perp in exchange.markets: lg.warning(f"{constants.NEON_YELLOW}Did you mean '{perp_sym_perp}'?{constants.RESET}")
                 elif possible_matches: lg.warning(f"{constants.NEON_YELLOW}Possible matches found: {possible_matches[:5]}{constants.RESET}")
             return None

        market = exchange.market(symbol)
        if market:
            market_type = market.get('type', 'unknown')
            is_contract = market.get('contract', False) or market_type in ['swap', 'future']
            contract_type = "N/A"
            if is_contract:
                if market.get('linear'): contract_type = "Linear"
                elif market.get('inverse'): contract_type = "Inverse"
                else: contract_type = "Unknown Contract"

            lg.debug(f"Market Info for {symbol}: ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                     f"Type={market_type}, IsContract={is_contract}, ContractType={contract_type}, "
                     f"Precision(P/A): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}, "
                     f"Limits(Amt Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                     f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                     f"Contract Size: {market.get('contractSize', 'N/A')}")
            market['is_contract'] = is_contract
            return market
        else:
             lg.error(f"{constants.NEON_RED}Market dictionary unexpectedly not found for validated symbol {symbol}.{constants.RESET}"); return None

    except ccxt.BadSymbol as e: lg.error(f"{constants.NEON_RED}Symbol '{symbol}' is invalid or not supported by {exchange.id}: {e}{constants.RESET}"); return None
    except ccxt.NetworkError as e: lg.error(f"{constants.NEON_RED}Network error getting market info for {symbol}: {e}{constants.RESET}"); return None
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error getting market info for {symbol}: {e}{constants.RESET}", exc_info=True); return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: Dict, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions."""
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        fetch_all = False

        if exchange.has.get('fetchPositions'):
            try:
                market_id = market_info.get('id')
                if not market_id: lg.error(f"Cannot fetch position: Market ID missing for {symbol}."); return None
                positions = exchange.fetch_positions([market_id])
                lg.debug(f"Fetched single symbol position data for {symbol} (ID: {market_id}). Count: {len(positions)}")
            except ccxt.ArgumentsRequired: lg.debug(f"fetchPositions for {exchange.id} requires no arguments. Fetching all."); fetch_all = True
            except ccxt.ExchangeError as e:
                 no_pos_codes_v5 = [110025]; no_pos_messages = ["position not found", "position is closed", "no position found"]
                 err_str = str(e).lower(); bybit_code = getattr(e, 'code', None)
                 if any(msg in err_str for msg in no_pos_messages) or (bybit_code in no_pos_codes_v5):
                      lg.info(f"No position found for {symbol} (Exchange confirmed: {e})."); return None
                 else: lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=False); fetch_all = True
            except Exception as e: lg.error(f"Error fetching single position for {symbol}: {e}", exc_info=True); fetch_all = True
        else: lg.warning(f"Exchange {exchange.id} does not support fetchPositions. Cannot check position status."); return None

        if fetch_all:
            lg.debug(f"Attempting to fetch all positions for {exchange.id}...")
            try:
                 all_positions = exchange.fetch_positions()
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
            except Exception as e: lg.error(f"Error fetching all positions for {symbol}: {e}", exc_info=True); return None

        active_position = None
        size_threshold = Decimal('1e-9')
        try:
            amount_prec_val = market_info.get('precision', {}).get('amount')
            if amount_prec_val is not None:
                if isinstance(amount_prec_val, (float, str)):
                     amount_step = Decimal(str(amount_prec_val))
                     if amount_step > 0: size_threshold = amount_step / Decimal('2')
                elif isinstance(amount_prec_val, int):
                     if amount_prec_val >= 0: size_threshold = Decimal('1e-' + str(amount_prec_val + 1))
        except Exception as thresh_err: lg.warning(f"Could not determine size threshold from market precision ({thresh_err}). Using default: {size_threshold}")
        lg.debug(f"Using position size threshold: {size_threshold} for {symbol}")

        for pos in positions:
            pos_size_str = None
            if pos.get('contracts') is not None: pos_size_str = str(pos['contracts'])
            elif isinstance(pos.get('info'), dict) and pos['info'].get('size') is not None: pos_size_str = str(pos['info']['size'])
            if pos_size_str is None: lg.debug(f"Skipping position entry, could not determine size: {pos}"); continue
            try:
                position_size = Decimal(pos_size_str)
                if abs(position_size) > size_threshold:
                    active_position = pos; lg.debug(f"Found potential active position entry for {symbol} with size {position_size}."); break
            except (InvalidOperation, ValueError, TypeError) as parse_err: lg.warning(f"Could not parse position size '{pos_size_str}' for {symbol}: {parse_err}"); continue

        if active_position:
            try:
                size_decimal = Decimal(str(active_position.get('contracts', active_position.get('info',{}).get('size', '0'))))
                active_position['contractsDecimal'] = size_decimal
                side = active_position.get('side')
                if side not in ['long', 'short']:
                    if size_decimal > size_threshold: side = 'long'
                    elif size_decimal < -size_threshold: side = 'short'
                    else: lg.warning(f"Position size {size_decimal} near zero for {symbol}, cannot reliably determine side."); return None
                    active_position['side'] = side; lg.debug(f"Inferred position side as '{side}' based on size {size_decimal}.")

                entry_price_str = active_position.get('entryPrice') or active_position.get('info', {}).get('avgPrice')
                if entry_price_str: active_position['entryPriceDecimal'] = Decimal(str(entry_price_str))
                else: active_position['entryPriceDecimal'] = None

                info_dict = active_position.get('info', {})
                if active_position.get('stopLossPrice') is None: active_position['stopLossPrice'] = info_dict.get('stopLoss')
                if active_position.get('takeProfitPrice') is None: active_position['takeProfitPrice'] = info_dict.get('takeProfit')
                active_position['trailingStopLossValue'] = info_dict.get('trailingStop')
                active_position['trailingStopActivationPrice'] = info_dict.get('activePrice')
                timestamp_ms = active_position.get('timestamp') or info_dict.get('updatedTime')
                active_position['timestamp_ms'] = timestamp_ms

                def format_log_val(val, is_price=True, is_size=False):
                     if val is None or str(val).strip() == '' or str(val) == '0': return 'N/A'
                     try:
                          d_val = Decimal(str(val))
                          if is_size:
                              amt_prec = 8
                              try:
                                  amt_prec_val = market_info['precision']['amount']
                                  if isinstance(amt_prec_val, (float, str)): amt_step = Decimal(str(amt_prec_val)); amt_prec = abs(amt_step.normalize().as_tuple().exponent) if amt_step > 0 else 8
                                  elif isinstance(amt_prec_val, int): amt_prec = amt_prec_val
                              except Exception: pass
                              return f"{abs(d_val):.{amt_prec}f}"
                          elif is_price:
                              price_prec = 6
                              try: price_prec = utils.get_price_precision(market_info, lg)
                              except Exception: pass
                              return f"{d_val:.{price_prec}f}"
                          else: return f"{d_val:.4f}"
                     except Exception: return str(val)

                entry_price_fmt = format_log_val(active_position.get('entryPriceDecimal'))
                contracts_fmt = format_log_val(size_decimal, is_size=True)
                liq_price_fmt = format_log_val(active_position.get('liquidationPrice'))
                leverage_str = active_position.get('leverage', info_dict.get('leverage'))
                leverage_fmt = f"{Decimal(str(leverage_str)):.1f}x" if leverage_str is not None else 'N/A'
                pnl_fmt = format_log_val(active_position.get('unrealizedPnl'), is_price=False)
                sl_price_fmt = format_log_val(active_position.get('stopLossPrice'))
                tp_price_fmt = format_log_val(active_position.get('takeProfitPrice'))
                tsl_dist_fmt = format_log_val(active_position.get('trailingStopLossValue'), is_price=False)
                tsl_act_fmt = format_log_val(active_position.get('trailingStopActivationPrice'))

                logger.info(f"{constants.NEON_GREEN}Active {side.upper()} position found ({symbol}):{constants.RESET} "
                            f"Size={contracts_fmt}, Entry={entry_price_fmt}, Liq={liq_price_fmt}, "
                            f"Lev={leverage_fmt}, PnL={pnl_fmt}, SL={sl_price_fmt}, TP={tp_price_fmt}, "
                            f"TSL(Dist/Act): {tsl_dist_fmt}/{tsl_act_fmt}")
                logger.debug(f"Full position details for {symbol}: {active_position}")
                return active_position

            except (InvalidOperation, ValueError, TypeError) as proc_err: lg.error(f"Error processing active position details for {symbol} (Decimal/Type Error): {proc_err}", exc_info=False); lg.debug(f"Problematic position data: {active_position}"); return None
            except Exception as proc_err: lg.error(f"Error processing active position details for {symbol}: {proc_err}", exc_info=True); lg.debug(f"Problematic position data: {active_position}"); return None
        else: logger.info(f"No active open position found for {symbol}."); return None

    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{constants.RESET}", exc_info=True)
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling exchange specifics (like Bybit V5)."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract: lg.info(f"Leverage setting skipped for {symbol} (Not a contract market)."); return True
    if not isinstance(leverage, int) or leverage <= 0: lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be a positive integer."); return False
    if not exchange.has.get('setLeverage'):
         if not exchange.has.get('setMarginMode'): lg.error(f"{constants.NEON_RED}Exchange {exchange.id} does not support setLeverage or setMarginMode via CCXT. Cannot set leverage.{constants.RESET}"); return False
         else: lg.warning(f"{constants.NEON_YELLOW}Exchange {exchange.id} uses setMarginMode for leverage. Attempting via setMarginMode...{constants.RESET}")

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        if 'bybit' in exchange.id.lower():
             leverage_str = str(leverage)
             params = {'buyLeverage': leverage_str, 'sellLeverage': leverage_str}
             lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")
        lg.info(f"{constants.NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Check position details for confirmation).{constants.RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower(); exchange_code = getattr(e, 'code', None)
        lg.error(f"{constants.NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {exchange_code}){constants.RESET}")
        if 'bybit' in exchange.id.lower():
             if exchange_code == 110043 or "leverage not modified" in err_str: lg.info(f"{constants.NEON_YELLOW}Leverage for {symbol} likely already set to {leverage}x (Exchange confirmation).{constants.RESET}"); return True
             elif exchange_code in [110028, 110009, 110055] or "margin mode" in err_str: lg.error(f"{constants.NEON_YELLOW} >> Hint: Check Margin Mode (Isolated/Cross) compatibility with leverage setting. May need to set margin mode first.{constants.RESET}")
             elif exchange_code == 110044 or "risk limit" in err_str: lg.error(f"{constants.NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed the current risk limit tier for {symbol}. Check Bybit Risk Limits.{constants.RESET}")
             elif exchange_code == 110013 or "parameter error" in err_str: lg.error(f"{constants.NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid (e.g., too high) for {symbol}. Check allowed range.{constants.RESET}")
             elif "set margin mode" in err_str: lg.error(f"{constants.NEON_YELLOW} >> Hint: Operation might require setting margin mode first/again using `set_margin_mode`.{constants.RESET}")
    except ccxt.NetworkError as e: lg.error(f"{constants.NEON_RED}Network error setting leverage for {symbol}: {e}{constants.RESET}")
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error setting leverage for {symbol}: {e}{constants.RESET}", exc_info=True)
    return False


def place_trade(
    exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal,
    market_info: Dict, logger: Optional[logging.Logger] = None, order_type: str = 'market',
    limit_price: Optional[Decimal] = None, reduce_only: bool = False, params: Optional[Dict] = None,
    quote_currency: str = 'USDT' # Pass quote currency for logging
) -> Optional[Dict]:
    """Places an order (market or limit) using CCXT."""
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    amount_for_api: Optional[float] = None
    price_for_api: Optional[float] = None
    price_str_formatted: Optional[str] = None

    try:
        size_str_formatted = exchange.amount_to_precision(symbol, f"{position_size:.{getcontext().prec}f}")
        amount_decimal = Decimal(size_str_formatted)
        if amount_decimal <= 0: lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Invalid or zero position size after formatting ({amount_decimal}). Original: {position_size}"); return None
        amount_for_api = float(size_str_formatted)
    except (ccxt.ExchangeError, InvalidOperation, ValueError, TypeError) as e: lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Failed to format/convert size {position_size}: {e}"); return None

    if order_type == 'limit':
        if limit_price is None or not isinstance(limit_price, Decimal) or limit_price <= 0: lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Limit order requested but invalid limit_price ({limit_price}) provided."); return None
        try:
            price_str_formatted = exchange.price_to_precision(symbol, float(limit_price))
            price_for_api = float(price_str_formatted)
            if price_for_api <= 0: raise ValueError("Formatted limit price is non-positive")
        except (ccxt.ExchangeError, ValueError, TypeError) as e: lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Failed to format/validate limit price {limit_price}: {e}"); return None

    order_params = {'reduceOnly': reduce_only}
    if params:
         external_params = {k: v for k, v in params.items() if k not in order_params}
         order_params.update(external_params)
    if reduce_only and order_type == 'market': order_params['timeInForce'] = 'IOC'

    log_price = f"Limit @ {price_str_formatted}" if order_type == 'limit' and price_str_formatted else "Market"
    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type.upper()} order for {symbol}:")
    lg.info(f"  Size: {amount_for_api} {size_unit}")
    if order_type == 'limit' and price_str_formatted: lg.info(f"  Limit Price: {price_str_formatted}")
    lg.info(f"  ReduceOnly: {reduce_only}"); lg.info(f"  Params: {order_params}")

    try:
        order: Optional[Dict] = None
        if order_type == 'market':
            order = exchange.create_order(symbol=symbol, type='market', side=side, amount=amount_for_api, price=None, params=order_params)
        elif order_type == 'limit':
            order = exchange.create_order(symbol=symbol, type='limit', side=side, amount=amount_for_api, price=price_for_api, params=order_params)
        else: lg.error(f"Unsupported order type '{order_type}' in place_trade function."); return None

        if order:
            order_id = order.get('id', 'N/A'); order_status = order.get('status', 'N/A')
            filled_amount_raw = order.get('filled'); filled_amount = float(filled_amount_raw) if filled_amount_raw is not None else 0.0
            avg_price = order.get('average')
            lg.info(f"{constants.NEON_GREEN}{action_desc} Trade Placed Successfully!{constants.RESET}")
            lg.info(f"  Order ID: {order_id}, Initial Status: {order_status}")
            if filled_amount is not None and filled_amount > 0: lg.info(f"  Filled Amount: {filled_amount}")
            if avg_price: lg.info(f"  Average Fill Price: {avg_price}")
            lg.debug(f"Raw order response ({symbol} {side} {action_desc}): {order}")
            return order
        else: lg.error(f"{constants.NEON_RED}Order placement call returned None without raising an exception for {symbol}.{constants.RESET}"); return None

    except ccxt.InsufficientFunds as e:
        lg.error(f"{constants.NEON_RED}Insufficient funds to place {side} {order_type} order ({symbol}): {e}{constants.RESET}")
        try: balance = fetch_balance(exchange, quote_currency, lg); lg.info(f"Current Balance: {balance} {quote_currency}")
        except: pass
    except ccxt.InvalidOrder as e:
        lg.error(f"{constants.NEON_RED}Invalid order parameters for {side} {order_type} order ({symbol}): {e}{constants.RESET}")
        lg.error(f"  > Used Parameters: amount={amount_for_api}, price={price_for_api if order_type=='limit' else 'N/A'}, params={order_params}")
        if "Order price is not following the tick size" in str(e): lg.error("  >> Hint: Check limit_price alignment with market tick size.")
        if "Order size is not following the step size" in str(e): lg.error("  >> Hint: Check position_size alignment with market amount step size.")
        if "minNotional" in str(e) or "cost" in str(e).lower() or "minimum value" in str(e).lower(): lg.error("  >> Hint: Order cost might be below the minimum required by the exchange.")
        exchange_code = getattr(e, 'code', None)
        if reduce_only and exchange_code == 110014: lg.error(f"{constants.NEON_YELLOW}  >> Hint (Bybit 110014): Reduce-only order failed. Position might be closed, size incorrect, or side wrong?{constants.RESET}")
    except ccxt.NetworkError as e: lg.error(f"{constants.NEON_RED}Network error placing {action_desc} order ({symbol}): {e}{constants.RESET}")
    except ccxt.ExchangeError as e:
        exchange_code = getattr(e, 'code', None)
        lg.error(f"{constants.NEON_RED}Exchange error placing {action_desc} order ({symbol}): {e} (Code: {exchange_code}){constants.RESET}")
        if reduce_only and exchange_code == 110025: lg.warning(f"{constants.NEON_YELLOW} >> Hint (Bybit 110025): Position might have been closed already when trying to place reduce-only order.{constants.RESET}")
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error placing {action_desc} order ({symbol}): {e}{constants.RESET}", exc_info=True)
    return None


def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None,
) -> bool:
    """Internal helper to set SL, TP, or TSL for an existing position using Bybit's V5 API."""
    lg = logger
    if 'bybit' not in exchange.id.lower(): lg.error(f"Protection setting via private_post is currently implemented only for Bybit. Cannot set for {exchange.id}."); return False
    if not market_info.get('is_contract', False): lg.warning(f"Protection setting skipped for {symbol} (Not a contract market)."); return False
    if not position_info: lg.error(f"Cannot set protection for {symbol}: Missing position information."); return False

    pos_side = position_info.get('side')
    if pos_side not in ['long', 'short']: lg.error(f"Cannot set protection for {symbol}: Invalid or missing position side ('{pos_side}') in position_info."); return False
    position_idx = 0
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val); lg.debug(f"Using positionIdx: {position_idx} from position info for {symbol}.")
        else: lg.debug(f"positionIdx not found in position_info['info'] for {symbol}. Using default {position_idx} (assuming One-Way mode).")
    except Exception as idx_err: lg.warning(f"Could not parse positionIdx from position info ({idx_err}), using default {position_idx}.")

    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    has_tsl = (isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and
               isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0)

    initial_protection_requested = has_sl or has_tp or has_tsl
    if not initial_protection_requested: lg.info(f"No valid protection parameters provided for {symbol} (PosIdx: {position_idx}). No protection set/updated."); return True

    category = 'linear' if market_info.get('linear', True) else 'inverse'
    params = {
        'category': category, 'symbol': market_info['id'], 'tpslMode': 'Full',
        'slTriggerBy': 'LastPrice', 'tpTriggerBy': 'LastPrice', 'positionIdx': position_idx
    }
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} PosIdx: {position_idx}):"]

    try:
        price_precision = utils.get_price_precision(market_info, lg)
        min_tick = utils.get_min_tick_size(market_info, lg)

        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not isinstance(price_decimal, Decimal) or price_decimal <= 0: return None
            try: return exchange.price_to_precision(symbol, float(price_decimal))
            except Exception as e: lg.warning(f"Failed to format price {price_decimal} using price_to_precision: {e}. Price will not be set."); return None

        formatted_tsl_distance = None; formatted_activation_price = None
        if has_tsl:
            try:
                 dist_prec = abs(min_tick.normalize().as_tuple().exponent) if min_tick > 0 else price_precision
                 formatted_tsl_distance = exchange.decimal_to_precision(trailing_stop_distance, exchange.ROUND, precision=dist_prec, padding_mode=exchange.NO_PADDING)
                 if min_tick > 0 and Decimal(formatted_tsl_distance) < min_tick: lg.warning(f"Calculated TSL distance {formatted_tsl_distance} is less than min tick {min_tick}. Adjusting to min tick."); formatted_tsl_distance = str(min_tick)
            except (ccxt.ExchangeError, InvalidOperation, ValueError, TypeError) as e: lg.warning(f"Failed to format TSL distance {trailing_stop_distance} using decimal_to_precision: {e}. TSL distance will not be set."); formatted_tsl_distance = None
            formatted_activation_price = format_price(tsl_activation_price)
            if formatted_tsl_distance and formatted_activation_price and Decimal(formatted_tsl_distance) > 0:
                params['trailingStop'] = formatted_tsl_distance; params['activePrice'] = formatted_activation_price
                log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                has_sl = False; lg.debug("TSL parameters added. Fixed SL will be ignored by the exchange.")
            else: lg.error(f"Failed to format valid TSL parameters for {symbol}. TSL will not be set."); has_tsl = False

        if has_sl:
            formatted_sl = format_price(stop_loss_price)
            if formatted_sl: params['stopLoss'] = formatted_sl; log_parts.append(f"  Fixed SL: {formatted_sl}")
            else: has_sl = False

        if has_tp:
            formatted_tp = format_price(take_profit_price)
            if formatted_tp: params['takeProfit'] = formatted_tp; log_parts.append(f"  Fixed TP: {formatted_tp}")
            else: has_tp = False

    except Exception as fmt_err: lg.error(f"Error during formatting/preparation of protection parameters for {symbol}: {fmt_err}", exc_info=True); return False

    if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
        lg.warning(f"No valid protection parameters could be formatted or remained after adjustments for {symbol} (PosIdx: {position_idx}). No API call made.")
        return False if initial_protection_requested else True

    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call: exchange.private_post('/v5/position/set-trading-stop', params={params})")

    try:
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', 'Unknown Error'); ret_ext = response.get('retExtInfo', {})

        if ret_code == 0:
            if "not modified" in ret_msg.lower(): lg.info(f"{constants.NEON_YELLOW}Position protection already set to target values or only partially modified for {symbol} (PosIdx: {position_idx}). Response: {ret_msg}{constants.RESET}")
            else: lg.info(f"{constants.NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol} (PosIdx: {position_idx}).{constants.RESET}")
            return True
        else:
            lg.error(f"{constants.NEON_RED}Failed to set protection for {symbol} (PosIdx: {position_idx}): {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{constants.RESET}")
            if ret_code == 110013: lg.error(f"{constants.NEON_YELLOW} >> Hint (110013 - Parameter Error): Check SL/TP prices vs entry price, TSL distance/activation validity, tick size compliance, tpslMode.{constants.RESET}")
            elif ret_code == 110036: lg.error(f"{constants.NEON_YELLOW} >> Hint (110036 - TSL Price Invalid): TSL Activation price '{params.get('activePrice')}' likely invalid (already passed? wrong side? too close to current price?).{constants.RESET}")
            elif ret_code == 110086: lg.error(f"{constants.NEON_YELLOW} >> Hint (110086): Stop Loss price cannot be equal to Take Profit price.{constants.RESET}")
            elif ret_code == 110043:
                  if "leverage not modified" in ret_msg.lower(): lg.info(f"{constants.NEON_YELLOW} >> Hint (110043 - Leverage Not Modified): Leverage likely already set. Protection setting might still have failed due to other reasons.{constants.RESET}")
                  else: lg.error(f"{constants.NEON_YELLOW} >> Hint (110043): Position status prevents modification (e.g., during liquidation?).{constants.RESET}")
            elif ret_code == 110025: lg.error(f"{constants.NEON_YELLOW} >> Hint (110025): Position may have closed before protection could be set, or positionIdx mismatch?{constants.RESET}")
            elif "trailing stop value invalid" in ret_msg.lower(): lg.error(f"{constants.NEON_YELLOW} >> Hint: Trailing Stop distance '{params.get('trailingStop')}' likely invalid (too small? too large? violates tick size rules?).{constants.RESET}")
            return False

    except ccxt.AuthenticationError as e: lg.error(f"{constants.NEON_RED}Authentication error during protection API call for {symbol}: {e}{constants.RESET}"); return False
    except ccxt.NetworkError as e: lg.error(f"{constants.NEON_RED}Network error during protection API call for {symbol}: {e}{constants.RESET}"); return False
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error during protection API call for {symbol}: {e}{constants.RESET}", exc_info=True); return False
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    config: Dict[str, Any], logger: logging.Logger, take_profit_price: Optional[Decimal] = None
) -> bool:
    """Calculates and sets Trailing Stop Loss using _set_position_protection (Bybit V5)."""
    lg = logger
    if not config.get("enable_trailing_stop", False): lg.info(f"Trailing Stop Loss is disabled in config for {symbol}. Skipping TSL setup."); return False

    try:
        callback_rate_str = config.get("trailing_stop_callback_rate", "0.005")
        activation_perc_str = config.get("trailing_stop_activation_percentage", "0.003")
        callback_rate = Decimal(str(callback_rate_str)); activation_percentage = Decimal(str(activation_perc_str))
    except (InvalidOperation, ValueError, TypeError) as e: lg.error(f"{constants.NEON_RED}Invalid TSL parameter format in config ({symbol}): {e}. Cannot calculate TSL.{constants.RESET}"); return False
    if callback_rate <= 0: lg.error(f"{constants.NEON_RED}Invalid 'trailing_stop_callback_rate' ({callback_rate}) in config. Must be positive for {symbol}.{constants.RESET}"); return False
    if activation_percentage < 0: lg.error(f"{constants.NEON_RED}Invalid 'trailing_stop_activation_percentage' ({activation_percentage}) in config. Cannot be negative for {symbol}.{constants.RESET}"); return False

    try:
        entry_price = position_info.get('entryPriceDecimal')
        side = position_info.get('side')
        if entry_price is None or not isinstance(entry_price, Decimal) or entry_price <= 0: lg.error(f"{constants.NEON_RED}Missing or invalid entry price ({entry_price}) in position info for TSL calc ({symbol}).{constants.RESET}"); return False
        if side not in ['long', 'short']: lg.error(f"{constants.NEON_RED}Missing or invalid position side ('{side}') in position info for TSL calc ({symbol}).{constants.RESET}"); return False
    except Exception as e: lg.error(f"{constants.NEON_RED}Error accessing position info for TSL calculation ({symbol}): {e}.{constants.RESET}"); lg.debug(f"Position info received: {position_info}"); return False

    try:
        price_precision = utils.get_price_precision(market_info, lg)
        min_tick_size = utils.get_min_tick_size(market_info, lg)
        activation_price: Optional[Decimal] = None
        activation_offset = entry_price * activation_percentage
        from decimal import ROUND_UP, ROUND_DOWN # Import rounding modes

        if side == 'long':
            raw_activation = entry_price + activation_offset
            if min_tick_size > 0: activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
            else: rounding_factor = Decimal('1e-' + str(price_precision)); activation_price = raw_activation.quantize(rounding_factor, rounding=ROUND_UP)
            if activation_percentage > 0 and min_tick_size > 0 and activation_price <= entry_price: activation_price = ((entry_price + min_tick_size) / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size; lg.debug(f"Adjusted LONG TSL activation price to be at least one tick above entry: {activation_price}")
            elif activation_percentage == 0 and min_tick_size > 0: activation_price = ((entry_price + min_tick_size) / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size; lg.debug(f"Immediate TSL activation (0%) requested. Setting activation slightly above entry: {activation_price}")
        else: # short
            raw_activation = entry_price - activation_offset
            if min_tick_size > 0: activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
            else: rounding_factor = Decimal('1e-' + str(price_precision)); activation_price = raw_activation.quantize(rounding_factor, rounding=ROUND_DOWN)
            if activation_percentage > 0 and min_tick_size > 0 and activation_price >= entry_price: activation_price = ((entry_price - min_tick_size) / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size; lg.debug(f"Adjusted SHORT TSL activation price to be at least one tick below entry: {activation_price}")
            elif activation_percentage == 0 and min_tick_size > 0: activation_price = ((entry_price - min_tick_size) / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size; lg.debug(f"Immediate TSL activation (0%) requested. Setting activation slightly below entry: {activation_price}")

        if activation_price is None or activation_price <= 0: lg.error(f"{constants.NEON_RED}Calculated TSL activation price ({activation_price}) is invalid for {symbol}. Cannot set TSL.{constants.RESET}"); return False

        trailing_distance_raw = activation_price * callback_rate
        if min_tick_size > 0: trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        else: lg.warning("Min tick size is zero, cannot round trailing distance accurately."); trailing_distance = trailing_distance_raw
        if min_tick_size > 0 and trailing_distance < min_tick_size: lg.warning(f"Calculated TSL distance {trailing_distance} is smaller than min tick {min_tick_size}. Adjusting to min tick."); trailing_distance = min_tick_size
        if trailing_distance <= 0: lg.error(f"{constants.NEON_RED}Calculated TSL distance is zero or negative ({trailing_distance}) for {symbol}. Cannot set TSL.{constants.RESET}"); return False

        lg.info(f"Calculated TSL Parameters for {symbol} ({side.upper()}):")
        lg.info(f"  Entry={entry_price:.{price_precision}f}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  => Activation Price (Target): {activation_price:.{price_precision}f}")
        lg.info(f"  => Trailing Distance (Target): {trailing_distance:.{price_precision}f}")
        if isinstance(take_profit_price, Decimal) and take_profit_price > 0: lg.info(f"  Take Profit Price (Target): {take_profit_price:.{price_precision}f} (Will be set simultaneously)")
        else: lg.debug("  Take Profit: Not being set or updated with TSL.")

        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
            stop_loss_price=None, take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance, tsl_activation_price=activation_price
        )

    except (InvalidOperation, ValueError, TypeError) as e: lg.error(f"{constants.NEON_RED}Error calculating or preparing TSL parameters for {symbol} (Decimal/Type Error): {e}{constants.RESET}", exc_info=False); return False
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error calculating or preparing TSL parameters for {symbol}: {e}{constants.RESET}", exc_info=True); return False

```

```python
