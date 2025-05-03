# File: market_data.py
# -*- coding: utf-8 -*-

"""
Functions for Fetching Market Data from Bybit (Balances, OHLCV, Tickers, etc.)
"""

import logging
import sys
import time
import random
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Literal, Union

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas library not found. Install: pip install pandas")
    # Define dummy DataFrame if pandas is not available for basic checks
    class DummyDataFrame: pass
    pd = type('module', (object,), {'DataFrame': DummyDataFrame, 'to_datetime': lambda *args, **kwargs: None})()
try:
    from colorama import Fore, Style, Back
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored. Install: pip install colorama")
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

from config import Config
from utils import (retry_api_call, _get_v5_category, safe_decimal_conversion,
                   format_price, format_amount, send_sms_alert)

logger = logging.getLogger(__name__)


# Snippet 3 / Function 3: Fetch USDT Balance (V5 UNIFIED)
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_usdt_balance(exchange: ccxt.bybit, config: Config) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Fetches the USDT balance (Total Equity and Available Balance) using Bybit V5 UNIFIED account logic.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        config: The configuration object (used for USDT_SYMBOL).

    Returns:
        A tuple containing (total_equity, available_balance) as Decimals,
        or (None, None) if fetching fails or balance cannot be parsed.

    Raises:
        Reraises CCXT exceptions for the retry decorator.
    """
    func_name = "fetch_usdt_balance"
    logger.debug(f"[{func_name}] Fetching USDT balance (Bybit V5 UNIFIED Account)...")

    try:
        params = {'accountType': 'UNIFIED'} # V5 requires specifying account type
        balance_data = exchange.fetch_balance(params=params)
        # logger.debug(f"[{func_name}] Raw balance data: {balance_data}") # Verbose

        info = balance_data.get('info', {})
        result_list = info.get('result', {}).get('list', [])

        equity: Optional[Decimal] = None
        available: Optional[Decimal] = None
        account_type_found: str = "N/A"

        if result_list:
            # Find the UNIFIED account details within the list
            unified_account_info = next((acc for acc in result_list if acc.get('accountType') == 'UNIFIED'), None)

            if unified_account_info:
                account_type_found = "UNIFIED"
                # Total equity for the unified account
                equity = safe_decimal_conversion(unified_account_info.get('totalEquity'))

                # Find USDT details within the 'coin' list of the unified account
                coin_list = unified_account_info.get('coin', [])
                usdt_coin_info = next((coin for coin in coin_list if coin.get('coin') == config.USDT_SYMBOL), None)

                if usdt_coin_info:
                    # V5 fields: availableToWithdraw seems most reliable for actual available funds
                    avail_val = usdt_coin_info.get('availableToWithdraw') or \
                                usdt_coin_info.get('availableBalance') # Fallback if Withdraw isn't present
                    available = safe_decimal_conversion(avail_val)
                    if available is None:
                        logger.warning(f"[{func_name}] Found USDT entry in UNIFIED but could not parse available balance from V5 fields: {usdt_coin_info}")
                        available = Decimal("0.0") # Default to 0 if parsing fails
                else:
                    logger.warning(f"[{func_name}] USDT coin data not found within the UNIFIED account details. Assuming 0 available USDT.")
                    available = Decimal("0.0") # Assume zero if USDT entry is missing

            else:
                # Fallback if UNIFIED account type is not explicitly found (should be rare for V5 UTA)
                logger.warning(f"[{func_name}] 'UNIFIED' account type not found in V5 balance response list. Trying fallback to first account in list.")
                if len(result_list) >= 1:
                     first_account = result_list[0]; account_type_found = first_account.get('accountType', 'UNKNOWN')
                     logger.warning(f"[{func_name}] Using first account found: Type '{account_type_found}'")
                     equity = safe_decimal_conversion(first_account.get('totalEquity') or first_account.get('equity')) # Check common equity fields
                     coin_list = first_account.get('coin', [])
                     usdt_coin_info = next((coin for coin in coin_list if coin.get('coin') == config.USDT_SYMBOL), None)
                     if usdt_coin_info:
                         # Use availableBalance or walletBalance as fallback
                         avail_val = usdt_coin_info.get('availableBalance') or usdt_coin_info.get('walletBalance')
                         available = safe_decimal_conversion(avail_val)
                         if available is None: available = Decimal("0.0")
                     else: available = Decimal("0.0") # Assume 0 if USDT not found
                else:
                     logger.error(f"[{func_name}] Balance response list is empty. Cannot determine balance.")


        # If V5 structure parsing failed or didn't find UNIFIED, try standard CCXT keys as a last resort
        if equity is None or available is None:
            logger.debug(f"[{func_name}] V5 structure parsing failed or incomplete. Trying standard CCXT balance keys...")
            usdt_balance_std = balance_data.get(config.USDT_SYMBOL, {})
            if equity is None: equity = safe_decimal_conversion(usdt_balance_std.get('total'))
            if available is None: available = safe_decimal_conversion(usdt_balance_std.get('free'))

            if equity is not None and available is not None:
                account_type_found = "CCXT Standard Fallback"
                logger.warning(f"[{func_name}] Used CCXT standard balance keys as fallback.")
            else:
                 # Raise error if balance couldn't be parsed at all after all attempts
                 raise ValueError(f"Failed to parse USDT balance from both V5 structure (Found: {account_type_found}) and Standard CCXT keys.")

        # Ensure non-negative values and handle potential None after fallbacks
        final_equity = max(Decimal("0.0"), equity) if equity is not None else Decimal("0.0")
        final_available = max(Decimal("0.0"), available) if available is not None else Decimal("0.0")

        logger.info(f"[{func_name}] USDT Balance Fetched (Source: {account_type_found}): "
                    f"Equity = {final_equity:.4f}, Available = {final_available:.4f}")
        return final_equity, final_available

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] Error fetching/parsing balance: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator

    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert("[BybitHelper] CRITICAL: Failed fetch USDT balance!", config)
        return None, None


# Snippet 6 / Function 6: Fetch OHLCV with Pagination
def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    config: Config,
    since: Optional[int] = None,
    limit_per_req: int = 1000, # Bybit V5 max limit is 1000
    max_total_candles: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data for a symbol using pagination to handle limits.

    Converts the fetched data into a pandas DataFrame with proper indexing and
    data types, performing basic validation and cleaning (NaN handling).
    Uses internal retries for individual chunk fetches.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        timeframe: CCXT timeframe string (e.g., '1m', '5m', '1h', '1d').
        config: Configuration object.
        since: Optional starting timestamp (milliseconds UTC) to fetch data from.
               If None, fetches the most recent data.
        limit_per_req: Number of candles to fetch per API request (max 1000 for Bybit V5).
        max_total_candles: Optional maximum number of candles to retrieve in total.

    Returns:
        A pandas DataFrame containing the OHLCV data, indexed by UTC timestamp,
        or None if fetching or processing fails completely. Returns an empty DataFrame
        if no data is available for the period.
    """
    func_name = "fetch_ohlcv_paginated"
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}[{func_name}] The exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None

    try:
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        if limit_per_req > 1000:
            logger.warning(f"[{func_name}] Requested limit_per_req ({limit_per_req}) exceeds Bybit V5 max (1000). Clamping to 1000.")
            limit_per_req = 1000
        elif limit_per_req <= 0:
             logger.warning(f"[{func_name}] Invalid limit_per_req ({limit_per_req}). Setting to 1000.")
             limit_per_req = 1000

        # Determine category for V5 API call
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            # Default to linear if category cannot be determined, log warning
            logger.warning(f"[{func_name}] Could not determine category for {symbol}. Assuming 'linear' for OHLCV fetch. This might fail for Spot/Inverse markets.")
            category = 'linear'

        params = {'category': category}

        since_str = pd.to_datetime(since, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S UTC') if since else 'Most Recent'
        max_total_str = str(max_total_candles) if max_total_candles else 'Unlimited'
        logger.info(f"{Fore.BLUE}[{func_name}] Fetching {symbol} OHLCV ({timeframe}). "
                    f"Limit/Req: {limit_per_req}, Since: {since_str}, Max Total: {max_total_str}, Category: {category}{Style.RESET_ALL}")

        all_candles: List[list] = []
        current_since = since
        request_count = 0
        max_requests = float('inf')
        if max_total_candles and max_total_candles > 0:
            # Calculate max requests needed, rounding up
            max_requests = (max_total_candles + limit_per_req - 1) // limit_per_req

        while True: # Loop until break conditions are met
            # Check if max total candles reached
            if max_total_candles and len(all_candles) >= max_total_candles:
                logger.info(f"[{func_name}] Reached max_total_candles limit ({max_total_candles}). Fetch complete.")
                break
            # Check if max calculated requests reached
            if request_count >= max_requests:
                 logger.info(f"[{func_name}] Reached maximum calculated requests ({int(max_requests)}) based on max_total_candles. Fetch complete.")
                 break

            request_count += 1
            # Determine the limit for this specific request
            fetch_limit = limit_per_req
            if max_total_candles:
                remaining_needed = max_total_candles - len(all_candles)
                fetch_limit = min(limit_per_req, remaining_needed)
                if fetch_limit <= 0: # Should not happen if max_total_candles check above works
                     logger.debug(f"[{func_name}] Calculated fetch_limit is zero or negative. Breaking loop.")
                     break

            logger.debug(f"[{func_name}] Fetch Chunk #{request_count}: Since={current_since}, Limit={fetch_limit}, Params={params}")

            candles_chunk: Optional[List[list]] = None
            last_fetch_error: Optional[Exception] = None

            # Internal retry loop for fetching this specific chunk
            for attempt in range(config.RETRY_COUNT + 1): # +1 to allow final attempt
                try:
                    candles_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit, params=params)
                    last_fetch_error = None # Reset error on success
                    break # Exit retry loop on success
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded) as e:
                    last_fetch_error = e
                    if attempt < config.RETRY_COUNT:
                        retry_delay = config.RETRY_DELAY_SECONDS * (2 ** attempt) * (random.uniform(0.8, 1.2)) # Exponential backoff with jitter
                        logger.warning(f"{Fore.YELLOW}[{func_name}] API Error chunk #{request_count} (Try {attempt + 1}/{config.RETRY_COUNT}): {e}. Retrying in {retry_delay:.2f}s...{Style.RESET_ALL}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{Fore.RED}[{func_name}] API Error chunk #{request_count} failed after {config.RETRY_COUNT} retries: {e}. Aborting chunk fetch.{Style.RESET_ALL}")
                except ccxt.ExchangeError as e:
                    last_fetch_error = e
                    logger.error(f"{Fore.RED}[{func_name}] ExchangeError on chunk #{request_count}: {e}. Aborting chunk fetch.{Style.RESET_ALL}")
                    break # Don't retry logical exchange errors
                except Exception as e:
                    last_fetch_error = e
                    logger.error(f"[{func_name}] Unexpected error fetching chunk #{request_count}: {e}", exc_info=True)
                    break # Don't retry unknown errors

            # If an error occurred and wasn't resolved by retries
            if last_fetch_error:
                logger.error(f"{Fore.RED}[{func_name}] Failed to fetch chunk #{request_count} after {config.RETRY_COUNT + 1} attempts. Last Error: {last_fetch_error}{Style.RESET_ALL}")
                logger.warning(f"[{func_name}] Returning potentially incomplete data ({len(all_candles)} candles) due to fetch failure in pagination loop.")
                break # Exit the main pagination loop

            # Check if the exchange returned an empty list (no more data)
            if not candles_chunk:
                logger.debug(f"[{func_name}] No more candles returned by exchange (Chunk #{request_count}). End of data for this range.")
                break # Exit main loop

            # Filter potential overlap: if the first candle of the new chunk has the same or earlier timestamp
            # than the last candle we already have, remove it and any earlier ones.
            if all_candles and candles_chunk[0][0] <= all_candles[-1][0]:
                first_new_ts = candles_chunk[0][0]
                last_old_ts = all_candles[-1][0]
                logger.warning(f"{Fore.YELLOW}[{func_name}] Overlap detected chunk #{request_count} (New starts at {first_new_ts} <= Last fetched {last_old_ts}). Filtering overlap.{Style.RESET_ALL}")
                candles_chunk = [c for c in candles_chunk if c[0] > last_old_ts]
                if not candles_chunk:
                    logger.debug(f"[{func_name}] Entire chunk #{request_count} was overlap or empty after filtering. End of data.")
                    break # Exit main loop

            logger.debug(f"[{func_name}] Fetched {len(candles_chunk)} new, non-overlapping candles (Chunk #{request_count}). Total collected: {len(all_candles) + len(candles_chunk)}")
            all_candles.extend(candles_chunk)

            # If the exchange returned fewer candles than requested, it usually means the end of available data
            if len(candles_chunk) < fetch_limit:
                logger.debug(f"[{func_name}] Received fewer candles ({len(candles_chunk)}) than requested ({fetch_limit}). Assuming end of available data.")
                break # Exit main loop

            # Prepare 'since' for the next request: timestamp of the last received candle + 1 timeframe interval
            current_since = candles_chunk[-1][0] + timeframe_ms

            # Respect rate limit (use exchange's value if available, else small delay)
            rate_limit_delay = exchange.rateLimit / 1000.0 if exchange.rateLimit and exchange.rateLimit > 0 else 0.1 # Default 100ms
            time.sleep(rate_limit_delay)

        # After the loop, process the collected candles
        return _process_ohlcv_list(all_candles, func_name, symbol, timeframe, max_total_candles)

    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
         logger.error(f"{Fore.RED}[{func_name}] Initial setup error for OHLCV fetch ({symbol}, {timeframe}): {e}{Style.RESET_ALL}")
         return None
    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error during OHLCV pagination setup: {e}{Style.RESET_ALL}", exc_info=True)
        return None

def _process_ohlcv_list(
    candle_list: List[list], parent_func_name: str, symbol: str, timeframe: str, max_candles: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Internal helper to convert OHLCV list to validated pandas DataFrame."""
    func_name = f"{parent_func_name}._process_ohlcv_list"
    cols = ['open', 'high', 'low', 'close', 'volume'] # Standard OHLCV columns

    if not candle_list:
        logger.warning(f"{Fore.YELLOW}[{func_name}] No candles were collected for {symbol} ({timeframe}). Returning empty DataFrame.{Style.RESET_ALL}")
        # Create an empty DataFrame with the correct structure and index type
        empty_df = pd.DataFrame(columns=cols).astype({c: float for c in cols})
        empty_df.index = pd.to_datetime([]).tz_localize('UTC') # Empty DatetimeIndex (UTC)
        empty_df.index.name = 'timestamp'
        return empty_df

    logger.debug(f"[{func_name}] Processing {len(candle_list)} raw candles for {symbol} ({timeframe})...")
    try:
        # Create DataFrame
        df = pd.DataFrame(candle_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to DatetimeIndex (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Remove rows where timestamp conversion failed
        if df.empty:
             raise ValueError("All timestamp conversions failed or resulted in NaT.")

        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric (Decimal or float based on pandas version/config)
        # Using float is generally faster for large datasets in pandas
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove duplicate timestamps (keep the first occurrence)
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            logger.debug(f"[{func_name}] Removed {initial_len - len(df)} duplicate timestamp entries.")

        # Check for and handle NaNs in OHLCV columns
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Found {total_nans} NaN values in OHLCV columns. Attempting forward fill... (Counts: {nan_counts[nan_counts > 0].to_dict()}){Style.RESET_ALL}")
            df.ffill(inplace=True) # Forward fill NaNs
            df.dropna(inplace=True) # Drop any remaining NaNs (e.g., at the start)
            if df.isnull().sum().sum() > 0:
                logger.error(f"{Fore.RED}[{func_name}] NaNs persisted after ffill and dropna! Check data source.{Style.RESET_ALL}")
                # Decide how to handle persistent NaNs - return None or the partial DataFrame?
                # return None

        # Sort by timestamp index
        df.sort_index(inplace=True)

        # Trim DataFrame if max_total_candles was specified and we exceeded it slightly
        if max_candles and len(df) > max_candles:
            logger.debug(f"[{func_name}] Trimming DataFrame from {len(df)} to the last {max_candles} candles.")
            df = df.iloc[-max_candles:]

        if df.empty:
            logger.error(f"{Fore.RED}[{func_name}] Processed DataFrame is empty after cleaning and validation for {symbol} ({timeframe}).{Style.RESET_ALL}")
            # Return the empty DF structure defined earlier
            empty_df = pd.DataFrame(columns=cols).astype({c: float for c in cols}); empty_df.index = pd.to_datetime([]).tz_localize('UTC'); empty_df.index.name = 'timestamp'
            return empty_df

        # Use logger.info for success
        logger.info(f"{Fore.GREEN}[{func_name}] Successfully processed {len(df)} valid candles for {symbol} ({timeframe}).{Style.RESET_ALL}")
        return df

    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Error processing OHLCV list into DataFrame: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# Snippet 11 / Function 11: Fetch Funding Rate
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_funding_rate(exchange: ccxt.bybit, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches the current funding rate details for a perpetual swap symbol on Bybit V5.
    Returns Decimals for rates/prices.
    """
    func_name = "fetch_funding_rate"
    logger.debug(f"[{func_name}] Fetching funding rate for {symbol}...")

    try:
        market = exchange.market(symbol)
        # Ensure it's a swap market
        if not market.get('swap', False):
            logger.error(f"{Fore.RED}[{func_name}] Symbol '{symbol}' is not a swap market. Cannot fetch funding rate.{Style.RESET_ALL}")
            return None

        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']:
            logger.error(f"{Fore.RED}[{func_name}] Invalid category '{category}' determined for funding rate fetch ({symbol}). Expected 'linear' or 'inverse'.{Style.RESET_ALL}")
            return None

        params = {'category': category}
        logger.debug(f"[{func_name}] Calling fetch_funding_rate with params: {params}")

        # Fetch the funding rate using ccxt
        funding_rate_info = exchange.fetch_funding_rate(symbol, params=params)

        # Process the result, converting numeric strings to Decimals
        processed_fr: Dict[str, Any] = {
            'symbol': funding_rate_info.get('symbol'),
            'fundingRate': safe_decimal_conversion(funding_rate_info.get('fundingRate')),
            'fundingTimestamp': funding_rate_info.get('fundingTimestamp'), # Milliseconds
            'fundingDatetime': funding_rate_info.get('fundingDatetime'), # ISO8601 String
            'markPrice': safe_decimal_conversion(funding_rate_info.get('markPrice')),
            'indexPrice': safe_decimal_conversion(funding_rate_info.get('indexPrice')),
            'nextFundingTime': funding_rate_info.get('nextFundingTimestamp'), # Milliseconds
            'nextFundingDatetime': None, # Will populate below
            'info': funding_rate_info.get('info', {}) # Raw exchange response
        }

        # Validate crucial fields
        if processed_fr['fundingRate'] is None:
            logger.warning(f"[{func_name}] Could not parse 'fundingRate' for {symbol} from response: {funding_rate_info.get('fundingRate')}")
            # Decide if this is critical - maybe return None or proceed with None rate?

        # Format next funding time if available
        if processed_fr['nextFundingTime']:
            try:
                # Use pandas for robust datetime conversion and formatting
                next_dt = pd.to_datetime(processed_fr['nextFundingTime'], unit='ms', utc=True)
                processed_fr['nextFundingDatetime'] = next_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            except Exception as dt_err:
                logger.warning(f"[{func_name}] Could not format next funding datetime from timestamp {processed_fr['nextFundingTime']}: {dt_err}")
                processed_fr['nextFundingDatetime'] = "Error formatting"

        # Log summary
        rate = processed_fr.get('fundingRate')
        next_dt_str = processed_fr.get('nextFundingDatetime', "N/A")
        rate_str = f"{rate:.6%}" if rate is not None else "N/A"
        logger.info(f"[{func_name}] Funding Rate Fetched {symbol}: Current Rate = {rate_str}, Next Funding Time = {next_dt_str}")

        return processed_fr

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching funding rate for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching funding rate for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# Snippet 13 / Function 13: Fetch L2 Order Book (Validated)
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_l2_order_book_validated(
    exchange: ccxt.bybit, symbol: str, limit: int, config: Config
) -> Optional[Dict[str, List[Tuple[Decimal, Decimal]]]]:
    """
    Fetches the Level 2 order book for a symbol using Bybit V5 fetchOrderBook and validates the data.
    Returns bids and asks as lists of [price, amount] tuples using Decimals, sorted correctly.
    """
    func_name = "fetch_l2_order_book_validated"
    logger.debug(f"[{func_name}] Fetching L2 Order Book for {symbol} (Limit: {limit})...")

    if not hasattr(exchange, 'fetch_order_book') or not exchange.has.get('fetchOrderBook'):
        logger.error(f"{Fore.RED}[{func_name}] Exchange '{exchange.id}' does not support fetchOrderBook.{Style.RESET_ALL}")
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)

        params = {}
        # Determine V5 specific limit constraints based on category
        # From Bybit Docs (as of late 2023): Spot: 50, Linear/Inverse: 200, Option: 25
        max_limit = 200 # Default reasonable max
        if category:
            params = {'category': category}
            max_limit = {'spot': 50, 'linear': 200, 'inverse': 200, 'option': 25}.get(category, 200)
        else:
            logger.warning(f"[{func_name}] Cannot determine category for {symbol}. Fetching OB without category param, limit might be restricted by default.")

        # Clamp the requested limit
        effective_limit = limit
        if limit > max_limit:
            logger.warning(f"[{func_name}] Requested limit {limit} exceeds max {max_limit} for category '{category or 'Unknown'}'. Clamping to {max_limit}.")
            effective_limit = max_limit
        elif limit <= 0:
            logger.warning(f"[{func_name}] Invalid limit {limit}. Using default depth (check config or use a reasonable value like 25).")
            effective_limit = config.ORDER_BOOK_FETCH_LIMIT # Use default from config

        logger.debug(f"[{func_name}] Calling fetchOrderBook with symbol='{symbol}', limit={effective_limit}, params={params}")
        order_book = exchange.fetch_order_book(symbol, limit=effective_limit, params=params)

        # Basic structure validation
        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book:
            logger.error(f"[{func_name}] Invalid order book structure received: {order_book}")
            raise ValueError("Invalid order book structure received from exchange.")
        if not isinstance(order_book['bids'], list) or not isinstance(order_book['asks'], list):
             logger.error(f"[{func_name}] Bids or asks are not lists: bids_type={type(order_book['bids'])}, asks_type={type(order_book['asks'])}")
             raise ValueError("Order book 'bids' or 'asks' are not lists.")

        # Process and validate bids/asks entries
        validated_bids: List[Tuple[Decimal, Decimal]] = []
        validated_asks: List[Tuple[Decimal, Decimal]] = []
        conversion_errors = 0

        # Process bids (price descending)
        for price_val, amount_val in order_book['bids']:
            price = safe_decimal_conversion(price_val)
            amount = safe_decimal_conversion(amount_val)
            if price is None or amount is None or price <= Decimal("0") or amount < Decimal("0"): # Allow zero amount? Bybit might return it.
                conversion_errors += 1
                # logger.debug(f"[{func_name}] Skipping invalid bid entry: Price={price_val}, Amount={amount_val}")
                continue
            validated_bids.append((price, amount))

        # Process asks (price ascending)
        for price_val, amount_val in order_book['asks']:
            price = safe_decimal_conversion(price_val)
            amount = safe_decimal_conversion(amount_val)
            if price is None or amount is None or price <= Decimal("0") or amount < Decimal("0"):
                conversion_errors += 1
                # logger.debug(f"[{func_name}] Skipping invalid ask entry: Price={price_val}, Amount={amount_val}")
                continue
            validated_asks.append((price, amount))

        if conversion_errors > 0:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Skipped {conversion_errors} invalid price/amount entries in order book for {symbol}.{Style.RESET_ALL}")

        # Crucial validation: Check for crossed book (best bid >= best ask)
        if validated_bids and validated_asks:
            best_bid = validated_bids[0][0] # Highest bid price
            best_ask = validated_asks[0][0] # Lowest ask price
            if best_bid >= best_ask:
                logger.error(f"{Fore.RED}[{func_name}] Order book is crossed for {symbol}! Best Bid ({best_bid}) >= Best Ask ({best_ask}). Data is unreliable.{Style.RESET_ALL}")
                # Depending on criticality, either return None or the potentially flawed data with a warning
                # return None # Fail hard on crossed book
                # Fall through for now, caller should be aware

        # Ensure sorting (CCXT usually guarantees this, but double check)
        # Bids should be descending by price, Asks ascending by price
        # validated_bids.sort(key=lambda x: x[0], reverse=True) # Uncomment if sorting needed
        # validated_asks.sort(key=lambda x: x[0])              # Uncomment if sorting needed

        logger.debug(f"[{func_name}] Successfully processed L2 Order Book for {symbol}. Valid Bids: {len(validated_bids)}, Valid Asks: {len(validated_asks)}")
        return {'bids': validated_bids, 'asks': validated_asks}

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] API/Validation Error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# Snippet 17 / Function 17: Fetch Ticker (Validated)
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_ticker_validated(
    exchange: ccxt.bybit, symbol: str, config: Config, max_age_seconds: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fetches the ticker for a symbol from Bybit V5, validates its freshness and key values.
    Returns a dictionary with Decimal values where appropriate.
    """
    func_name = "fetch_ticker_validated"
    logger.debug(f"[{func_name}] Fetching and validating ticker for {symbol} (Max Age: {max_age_seconds}s)...")

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)

        params = {}
        if category:
            params = {'category': category}
        else:
             logger.warning(f"[{func_name}] Cannot determine category for {symbol}. Fetching ticker without category param.")

        logger.debug(f"[{func_name}] Calling fetch_ticker with symbol='{symbol}', params={params}")
        ticker = exchange.fetch_ticker(symbol, params=params)

        # Validate timestamp (freshness)
        timestamp_ms = ticker.get('timestamp')
        current_time_ms = time.time() * 1000
        age_seconds = (current_time_ms - timestamp_ms) / 1000.0 if timestamp_ms else float('inf')

        if timestamp_ms is None:
             raise ValueError(f"Ticker for {symbol} has no timestamp.")
        if age_seconds > max_age_seconds:
             raise ValueError(f"Ticker for {symbol} is stale. Age: {age_seconds:.1f}s > Max: {max_age_seconds}s.")
        # Check for clocks significantly out of sync (e.g., ticker from the future)
        if age_seconds < -10: # Allow a small grace period for minor sync differences
             raise ValueError(f"Ticker timestamp {timestamp_ms} seems to be from the future (Current: {current_time_ms:.0f}, Diff: {age_seconds:.1f}s). Check system clock sync.")

        # Validate key price fields
        last_price = safe_decimal_conversion(ticker.get('last'))
        bid_price = safe_decimal_conversion(ticker.get('bid'))
        ask_price = safe_decimal_conversion(ticker.get('ask'))

        if last_price is None or last_price <= Decimal("0"):
             raise ValueError(f"Invalid or missing 'last' price in ticker: {ticker.get('last')}")
        # Bid/Ask might be None in some market states, log warning but don't necessarily fail
        if bid_price is None or bid_price <= Decimal("0"):
             logger.warning(f"[{func_name}] Ticker for {symbol} has invalid or missing 'bid' price: {ticker.get('bid')}")
        if ask_price is None or ask_price <= Decimal("0"):
             logger.warning(f"[{func_name}] Ticker for {symbol} has invalid or missing 'ask' price: {ticker.get('ask')}")

        # Calculate spread if possible and check validity
        spread: Optional[Decimal] = None
        spread_pct: Optional[Decimal] = None
        if bid_price and ask_price and bid_price > 0 and ask_price > 0:
             if bid_price >= ask_price:
                  # This is a critical error indicating bad data
                  raise ValueError(f"Ticker spread is invalid (crossed): Bid ({bid_price}) >= Ask ({ask_price})")
             spread = ask_price - bid_price
             try:
                 spread_pct = (spread / bid_price) * Decimal("100") if bid_price > 0 else Decimal("inf")
             except Exception: # Catch potential division issues if bid is extremely small
                 spread_pct = Decimal("inf")

        # Validate volume fields (allow zero, but not negative)
        base_volume = safe_decimal_conversion(ticker.get('baseVolume'))
        quote_volume = safe_decimal_conversion(ticker.get('quoteVolume'))
        if base_volume is not None and base_volume < Decimal("0"):
             logger.warning(f"[{func_name}] Ticker for {symbol} has negative baseVolume: {base_volume}. Treating as 0.")
             base_volume = Decimal("0.0")
        if quote_volume is not None and quote_volume < Decimal("0"):
             logger.warning(f"[{func_name}] Ticker for {symbol} has negative quoteVolume: {quote_volume}. Treating as 0.")
             quote_volume = Decimal("0.0")

        # Construct the validated ticker dictionary using Decimal where appropriate
        validated_ticker = {
            'symbol': ticker.get('symbol', symbol),
            'timestamp': timestamp_ms,
            'datetime': ticker.get('datetime'), # ISO8601 string
            'last': last_price,
            'bid': bid_price,
            'ask': ask_price,
            'bidVolume': safe_decimal_conversion(ticker.get('bidVolume')),
            'askVolume': safe_decimal_conversion(ticker.get('askVolume')),
            'baseVolume': base_volume, # 24h Base volume
            'quoteVolume': quote_volume, # 24h Quote volume
            'high': safe_decimal_conversion(ticker.get('high')), # 24h High
            'low': safe_decimal_conversion(ticker.get('low')), # 24h Low
            'open': safe_decimal_conversion(ticker.get('open')), # 24h Open price
            'close': last_price, # CCXT convention: close is same as last
            'change': safe_decimal_conversion(ticker.get('change')), # 24h absolute change
            'percentage': safe_decimal_conversion(ticker.get('percentage')), # 24h percentage change
            'average': safe_decimal_conversion(ticker.get('average')), # 24h average price
            'spread': spread, # Calculated spread value
            'spread_pct': spread_pct, # Calculated spread percentage
            'info': ticker.get('info', {}) # Raw exchange response
        }

        spread_str = f"{spread_pct:.4f}%" if spread_pct is not None and spread_pct != Decimal("inf") else "N/A"
        logger.debug(f"[{func_name}] Ticker validation OK for {symbol}: Last={format_price(exchange, symbol, last_price)}, "
                     f"Spread={spread_str} (Age: {age_seconds:.1f}s)")
        return validated_ticker

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] Failed to fetch or validate ticker for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching/validating ticker for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# Snippet 21 / Function 21: Fetch Recent Trades
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_recent_trades(
    exchange: ccxt.bybit, symbol: str, config: Config, limit: int = 100, min_size_filter: Optional[Decimal] = None
) -> Optional[List[Dict]]:
    """
    Fetches recent public trades for a symbol from Bybit V5, validates data,
    and returns a list of trade dictionaries with Decimal values, sorted recent first.
    Optionally filters trades smaller than `min_size_filter`.
    """
    func_name = "fetch_recent_trades"
    filter_log = f"(MinSize: {format_amount(exchange, symbol, min_size_filter) if min_size_filter else 'None'})"
    logger.debug(f"[{func_name}] Fetching recent trades for {symbol} (Limit: {limit}) {filter_log}...")

    # Validate and clamp limit
    # Bybit V5 limit for public trades is typically 1000
    max_trade_limit = 1000
    if limit > max_trade_limit:
        logger.warning(f"[{func_name}] Requested limit {limit} exceeds max {max_trade_limit}. Clamping to {max_trade_limit}.")
        limit = max_trade_limit
    elif limit <= 0:
        logger.warning(f"[{func_name}] Invalid limit {limit}. Using default of 100.")
        limit = 100

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)

        params = {}
        if category:
            params = {'category': category}
            # Potentially add baseCoin filter for spot if needed, or execType=Trade? Check API docs.
            # params['execType'] = 'Trade' # Might help filter non-trade events if API returns them
        else:
             logger.warning(f"[{func_name}] Cannot determine category for {symbol}. Fetching trades without category param.")

        logger.debug(f"[{func_name}] Calling fetch_trades with symbol='{symbol}', limit={limit}, params={params}")
        trades_raw = exchange.fetch_trades(symbol, limit=limit, params=params)

        if not trades_raw:
            logger.debug(f"[{func_name}] No recent trades found for {symbol} with current parameters.")
            return []

        processed_trades: List[Dict] = []
        conversion_errors = 0
        filtered_out_count = 0

        for trade in trades_raw:
            try:
                # Validate and convert core fields
                amount = safe_decimal_conversion(trade.get('amount'))
                price = safe_decimal_conversion(trade.get('price'))
                trade_id = trade.get('id')
                timestamp = trade.get('timestamp')
                side = trade.get('side') # Should be 'buy' or 'sell'

                # Check if essential fields are valid
                if not all([trade_id, timestamp, side, price, amount]) or price <= 0 or amount <= 0:
                    conversion_errors += 1
                    # logger.debug(f"[{func_name}] Skipping trade with invalid core data: {trade}")
                    continue

                # Apply size filter if specified
                if min_size_filter is not None and amount < min_size_filter:
                    filtered_out_count += 1
                    continue

                # Calculate cost, fallback if missing or clearly wrong
                cost = safe_decimal_conversion(trade.get('cost'))
                calculated_cost = price * amount
                # Allow small tolerance for rounding differences
                if cost is None or abs(cost - calculated_cost) > (calculated_cost * Decimal("0.001")): # 0.1% tolerance
                    if cost is not None:
                         # logger.debug(f"[{func_name}] Trade cost mismatch ({cost}) vs calculated ({calculated_cost}). Using calculated.")
                         pass # Logged if needed
                    cost = calculated_cost

                processed_trades.append({
                    'id': trade_id,
                    'timestamp': timestamp, # Milliseconds
                    'datetime': trade.get('datetime'), # ISO8601 string
                    'symbol': trade.get('symbol', symbol),
                    'side': side,
                    'price': price,
                    'amount': amount,
                    'cost': cost,
                    'takerOrMaker': trade.get('takerOrMaker'), # 'taker' or 'maker'
                    'info': trade.get('info', {}) # Raw exchange response for this trade
                })
            except Exception as proc_err:
                conversion_errors += 1
                logger.warning(f"{Fore.YELLOW}[{func_name}] Error processing individual trade: {proc_err}. Data: {trade}{Style.RESET_ALL}")

        if conversion_errors > 0:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Skipped {conversion_errors} trades due to processing/validation errors for {symbol}.{Style.RESET_ALL}")
        if filtered_out_count > 0:
            logger.debug(f"[{func_name}] Filtered out {filtered_out_count} trades smaller than {min_size_filter} for {symbol}.")

        # Sort trades by timestamp descending (most recent first) - CCXT usually returns this way, but good practice to ensure
        processed_trades.sort(key=lambda x: x['timestamp'], reverse=True)

        logger.info(f"[{func_name}] Fetched and processed {len(processed_trades)} valid trades for {symbol} {filter_log} (Limit: {limit}).")
        return processed_trades

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching trades for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching trades for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- END OF FILE market_data.py ---

# ---------------------------------------------------------------------------

