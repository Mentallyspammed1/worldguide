# File: market_data.py
# -*- coding: utf-8 -*-

"""
Functions for Fetching Market Data from Bybit (Balances, OHLCV, Tickers, etc.)

This module provides enhanced functions to interact with the Bybit API via the
CCXT library, focusing on robustness, clear logging, data validation, and
handling V5 API specifics (like categories and account types). It includes
features like automatic retries, pagination for historical data, and graceful
handling of potential API errors or missing optional libraries.
"""

import logging
import sys
import time
import random
from decimal import Decimal, InvalidOperation
from typing import Optional, Dict, List, Tuple, Any, Union

# --- Dependency Handling ---
try:
    import ccxt

    # Check for Bybit specific class, as ccxt is a meta-package
    if not hasattr(ccxt, "bybit"):
        print("Error: ccxt library is installed, but Bybit exchange support might be missing or outdated.")
        # Optionally add instructions to update ccxt or check installation.
        # sys.exit(1) # Might be too strict, let it fail later if used.
except ImportError:
    print("ERROR: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    print(
        "WARNING: pandas library not found. OHLCV data will be returned as lists, not DataFrames. Install: pip install pandas"
    )
    PANDAS_AVAILABLE = False

    # Define dummy DataFrame and related functions if pandas is not available
    # This allows type hinting and basic checks to pass without pandas.
    class DummyDataFrame:
        def __init__(self, data=None, columns=None, index=None):
            pass

        def is_empty(self):
            return True  # Simplification

        def dropna(self, **kwargs):
            return self

        def ffill(self, **kwargs):
            return self

        def sort_index(self, **kwargs):
            return self

        def iloc(self, *args, **kwargs):
            return self

        def isnull(self):
            return self  # Needs more sophisticated dummy if used

        def sum(self):
            return 0

        def astype(self, *args, **kwargs):
            return self

        index = None

    class DummyDateTimeIndex:
        name = None

        def tz_localize(self, *args, **kwargs):
            return self

    class DummyPandasModule:
        DataFrame = DummyDataFrame

        @staticmethod
        def to_datetime(*args, **kwargs):
            # Basic dummy: return input if int/float, else None.
            # A real application might need a minimal datetime conversion here.
            if isinstance(args[0], (int, float)):
                return args[0]  # Keep timestamp as number if pandas not available
            return None

        @staticmethod
        def isnull(val):
            return val is None

    pd = DummyPandasModule()


try:
    from colorama import Fore, Style, Back, init

    init(autoreset=True)  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    print("WARNING: colorama library not found. Logs will not be colored. Install: pip install colorama")
    COLORAMA_AVAILABLE = False

    # Dummy color class if colorama is not available
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""  # Return empty string for any color attribute

    Fore = Style = Back = DummyColor()

# --- Local Imports ---
try:
    from config import Config
    from utils import (
        retry_api_call,
        _get_v5_category,
        safe_decimal_conversion,
        format_price,
        format_amount,
        send_sms_alert,
    )
except ImportError as e:
    print(f"ERROR: Failed to import local modules (config, utils): {e}")
    print("Please ensure config.py and utils.py are in the same directory or accessible in PYTHONPATH.")
    sys.exit(1)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Basic logging configuration (can be overridden by external setup)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ============================================================================
# Function 3: Fetch USDT Balance (V5 UNIFIED)
# ============================================================================
@retry_api_call(max_retries=3, initial_delay=1.0, caught_exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
def fetch_usdt_balance(exchange: ccxt.bybit, config: Config) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Fetches the USDT balance (Total Equity and Available Balance) using Bybit V5 UNIFIED account logic.

    Prioritizes V5 UNIFIED account structure, then falls back to standard CCXT keys if necessary.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        config: The configuration object (used for USDT_SYMBOL).

    Returns:
        A tuple containing (total_equity, available_balance) as Decimals,
        or (None, None) if fetching fails critically after retries.
        Returns (Decimal('0.0'), Decimal('0.0')) if USDT is not found but the call succeeds.

    Raises:
        Re-raises ccxt.NetworkError, ccxt.ExchangeError for the retry decorator.
        ValueError: If the balance response structure is unexpected or parsing fails completely.
    """
    func_name = "fetch_usdt_balance"
    logger.debug(f"[{func_name}] Fetching USDT balance (Targeting Bybit V5 UNIFIED Account)...")

    try:
        params = {"accountType": "UNIFIED"}  # V5 requires specifying account type
        balance_data = exchange.fetch_balance(params=params)

        if not balance_data or "info" not in balance_data:
            logger.error(f"{Fore.RED}[{func_name}] Received invalid or empty balance response.{Style.RESET_ALL}")
            raise ValueError("Invalid or empty balance response received from exchange.")

        # --- V5 UNIFIED Parsing Logic ---
        info = balance_data.get("info", {})
        result_list = info.get("result", {}).get("list", [])

        equity_v5: Optional[Decimal] = None
        available_v5: Optional[Decimal] = None
        account_type_found: str = "N/A"
        usdt_symbol = config.USDT_SYMBOL  # e.g., 'USDT'

        if result_list:
            # Find the UNIFIED account details within the list
            unified_account_info = next((acc for acc in result_list if acc.get("accountType") == "UNIFIED"), None)

            if unified_account_info:
                account_type_found = "UNIFIED (V5)"
                logger.debug(f"[{func_name}] Found UNIFIED account section in V5 response.")
                # Total equity for the unified account
                equity_v5 = safe_decimal_conversion(unified_account_info.get("totalEquity"))

                # Find USDT details within the 'coin' list of the unified account
                coin_list = unified_account_info.get("coin", [])
                usdt_coin_info = next((coin for coin in coin_list if coin.get("coin") == usdt_symbol), None)

                if usdt_coin_info:
                    # V5 fields: availableToWithdraw seems most reliable for actual available funds
                    # Fallback to availableBalance if the primary field is missing
                    avail_val_withdraw = usdt_coin_info.get("availableToWithdraw")
                    avail_val_balance = usdt_coin_info.get("availableBalance")
                    available_v5 = safe_decimal_conversion(avail_val_withdraw)

                    if available_v5 is None:
                        logger.debug(
                            f"[{func_name}] 'availableToWithdraw' not found or invalid ({avail_val_withdraw}), trying 'availableBalance' ({avail_val_balance})."
                        )
                        available_v5 = safe_decimal_conversion(avail_val_balance)

                    if available_v5 is None:
                        logger.warning(
                            f"{Fore.YELLOW}[{func_name}] Found USDT entry in UNIFIED account, but could not parse available balance from V5 fields ('availableToWithdraw', 'availableBalance'): {usdt_coin_info}{Style.RESET_ALL}"
                        )
                        available_v5 = Decimal("0.0")  # Default to 0 if parsing fails within USDT entry
                else:
                    logger.info(
                        f"[{func_name}] {usdt_symbol} coin data not found within the UNIFIED account details. Assuming 0 available {usdt_symbol}."
                    )
                    available_v5 = Decimal("0.0")  # Assume zero if USDT entry is missing

            else:
                # Fallback if UNIFIED account type is not explicitly found
                logger.warning(
                    f"[{func_name}] 'UNIFIED' account type not explicitly found in V5 balance response list ({len(result_list)} accounts found). Attempting fallback using first account."
                )
                if result_list:  # Check again if list is not empty
                    first_account = result_list[0]
                    account_type_found = first_account.get("accountType", "UNKNOWN") + " (Fallback)"
                    logger.warning(f"[{func_name}] Using first account found: Type '{account_type_found}'")
                    # Try common equity fields
                    equity_v5 = safe_decimal_conversion(first_account.get("totalEquity") or first_account.get("equity"))
                    coin_list = first_account.get("coin", [])
                    usdt_coin_info = next((coin for coin in coin_list if coin.get("coin") == usdt_symbol), None)
                    if usdt_coin_info:
                        # Use availableBalance or walletBalance as fallback in non-UNIFIED structure
                        avail_val = usdt_coin_info.get("availableBalance") or usdt_coin_info.get("walletBalance")
                        available_v5 = safe_decimal_conversion(avail_val)
                        if available_v5 is None:
                            available_v5 = Decimal("0.0")
                    else:
                        logger.warning(
                            f"[{func_name}] {usdt_symbol} coin data not found within the fallback account ({account_type_found}). Assuming 0 available {usdt_symbol}."
                        )
                        available_v5 = Decimal("0.0")
                else:
                    logger.error(
                        f"{Fore.RED}[{func_name}] Balance response list is unexpectedly empty after initial check. Cannot determine balance.{Style.RESET_ALL}"
                    )
                    # Keep equity_v5 and available_v5 as None to trigger standard CCXT fallback

        else:
            logger.warning(
                f"[{func_name}] V5 balance response 'result.list' is empty or missing. Attempting standard CCXT fallback."
            )

        # --- Standard CCXT Fallback Logic ---
        # Try standard keys if V5 parsing failed or didn't yield results for either equity or available
        final_equity = equity_v5
        final_available = available_v5

        if final_equity is None or final_available is None:
            logger.info(
                f"[{func_name}] V5 structure parsing was incomplete (Equity Found: {final_equity is not None}, Available Found: {final_available is not None}). Trying standard CCXT balance keys..."
            )
            usdt_balance_std = balance_data.get(usdt_symbol, {})
            parsed_from_std = False

            if final_equity is None:
                equity_std = safe_decimal_conversion(usdt_balance_std.get("total"))
                if equity_std is not None:
                    final_equity = equity_std
                    parsed_from_std = True
                    logger.debug(f"[{func_name}] Parsed 'total' equity from standard CCXT keys.")

            if final_available is None:
                available_std = safe_decimal_conversion(usdt_balance_std.get("free"))
                if available_std is not None:
                    final_available = available_std
                    parsed_from_std = True
                    logger.debug(f"[{func_name}] Parsed 'free' (available) balance from standard CCXT keys.")

            if parsed_from_std:
                account_type_found = "CCXT Standard Keys (Fallback)"
                logger.warning(f"[{func_name}] Used standard CCXT balance keys as fallback for missing values.")
            elif final_equity is None or final_available is None:
                # If still None after all attempts
                logger.error(
                    f"{Fore.RED}[{func_name}] Failed to parse USDT balance from both V5 structure (Source: {account_type_found}) and Standard CCXT keys. Raw USDT data: {usdt_balance_std}{Style.RESET_ALL}"
                )
                raise ValueError(
                    f"Failed to parse required USDT balance fields (Equity: {final_equity}, Available: {final_available}) after all attempts."
                )

        # Ensure non-negative values and handle potential None remaining (shouldn't happen after ValueError above)
        final_equity = max(Decimal("0.0"), final_equity) if final_equity is not None else Decimal("0.0")
        final_available = max(Decimal("0.0"), final_available) if final_available is not None else Decimal("0.0")

        logger.info(
            f"{Fore.GREEN}[{func_name}] USDT Balance Fetched (Source: {account_type_found}): "
            f"Equity = {final_equity:.4f}, Available = {final_available:.4f}{Style.RESET_ALL}"
        )
        return final_equity, final_available

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        # Logged as warning because retry decorator might handle it
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] Network/Exchange Error fetching balance: {e}. Retrying if possible...{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator

    except (ValueError, KeyError, TypeError, InvalidOperation) as e:
        # Errors during parsing or unexpected structure
        logger.error(f"{Fore.RED}[{func_name}] Error processing balance data: {e}{Style.RESET_ALL}", exc_info=True)
        # Potentially send alert here if parsing failure is critical
        # send_sms_alert(f"[BybitHelper] WARNING: Failed to parse USDT balance: {e}", config)
        return None, None  # Indicate failure to the caller

    except Exception as e:
        # Catch-all for truly unexpected errors
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error fetching balance: {e}{Style.RESET_ALL}", exc_info=True
        )
        send_sms_alert("[BybitHelper] CRITICAL: Unexpected error fetching USDT balance!", config)
        return None, None


# ============================================================================
# Function 6: Fetch OHLCV with Pagination
# ============================================================================
# Note: Removing @retry_api_call here as the function implements its own robust
# internal retry logic per chunk, which is more suitable for pagination.
# Applying retry_api_call to the whole function could restart the entire
# pagination process on a single chunk failure after internal retries.
def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    config: Config,
    since: Optional[int] = None,
    limit_per_req: int = 1000,
    max_total_candles: Optional[int] = None,
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """
    Fetches historical OHLCV data for a symbol using pagination to handle limits.

    Uses internal retries for individual chunk fetches. Converts the fetched
    data into a pandas DataFrame if pandas is installed, otherwise returns a
    list of lists. Performs basic validation and cleaning (NaN handling if using pandas).

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        timeframe: CCXT timeframe string (e.g., '1m', '5m', '1h', '1d').
        config: Configuration object (used for retry settings).
        since: Optional starting timestamp (milliseconds UTC) to fetch data from.
               If None, fetches the most recent data backwards.
        limit_per_req: Number of candles to fetch per API request (max 1000 for Bybit V5).
        max_total_candles: Optional maximum number of candles to retrieve in total.

    Returns:
        - A pandas DataFrame containing the OHLCV data, indexed by UTC timestamp,
          if pandas is available.
        - A list of lists ([timestamp, open, high, low, close, volume]) if pandas
          is not available.
        - Returns an empty DataFrame/list if no data is available for the period.
        - Returns None if a critical setup error occurs (e.g., bad symbol).
    """
    func_name = "fetch_ohlcv_paginated"
    if not hasattr(exchange, "fetch_ohlcv") or not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}[{func_name}] The exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}"
        )
        return None

    try:
        # Validate timeframe and calculate duration in milliseconds
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    except (ValueError, TypeError) as e:
        logger.error(f"{Fore.RED}[{func_name}] Invalid timeframe '{timeframe}': {e}{Style.RESET_ALL}")
        return None

    # Validate and clamp limit_per_req
    bybit_v5_max_limit = 1000
    if not isinstance(limit_per_req, int) or limit_per_req <= 0:
        logger.warning(
            f"[{func_name}] Invalid limit_per_req ({limit_per_req}). Setting to default: {bybit_v5_max_limit}."
        )
        limit_per_req = bybit_v5_max_limit
    elif limit_per_req > bybit_v5_max_limit:
        logger.warning(
            f"[{func_name}] Requested limit_per_req ({limit_per_req}) exceeds Bybit V5 max ({bybit_v5_max_limit}). Clamping."
        )
        limit_per_req = bybit_v5_max_limit

    try:
        # Determine category for V5 API call
        market = exchange.market(symbol)  # Can raise BadSymbol
        category = _get_v5_category(market)
        if not category:
            # Default to linear if category cannot be determined, log warning
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Could not determine market category for {symbol}. Assuming 'linear' for OHLCV fetch. This might fail for Spot/Inverse markets.{Style.RESET_ALL}"
            )
            category = "linear"  # Bybit default for many operations if unspecified, but risky

        params = {"category": category}

        since_dt_str = (
            pd.to_datetime(since, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            if since and PANDAS_AVAILABLE
            else (str(since) if since else "Most Recent")
        )
        max_total_str = str(max_total_candles) if max_total_candles is not None else "Unlimited"
        logger.info(
            f"{Fore.BLUE}[{func_name}] Starting OHLCV fetch for {symbol} ({timeframe}). "
            f"Limit/Req: {limit_per_req}, Since: {since_dt_str}, Max Total: {max_total_str}, Category: {category}{Style.RESET_ALL}"
        )

        all_candles: List[list] = []
        current_since = since
        request_count = 0
        max_requests = float("inf")
        if max_total_candles is not None and max_total_candles > 0:
            # Calculate max requests needed, rounding up
            max_requests = (max_total_candles + limit_per_req - 1) // limit_per_req
            if max_requests <= 0:
                max_requests = 1  # Ensure at least one request if max_total > 0

        # --- Pagination Loop ---
        while True:
            # Check stopping conditions
            if max_total_candles is not None and len(all_candles) >= max_total_candles:
                logger.info(
                    f"[{func_name}] Reached or exceeded max_total_candles limit ({max_total_candles}). Fetch complete."
                )
                break
            if request_count >= max_requests:
                logger.info(
                    f"[{func_name}] Reached maximum calculated requests ({int(max_requests)}) based on max_total_candles. Fetch complete."
                )
                break

            request_count += 1
            # Determine the limit for this specific request, considering remaining needed
            fetch_limit = limit_per_req
            if max_total_candles is not None:
                remaining_needed = max_total_candles - len(all_candles)
                fetch_limit = min(limit_per_req, remaining_needed)
                if fetch_limit <= 0:  # Should be caught by the check at loop start, but safeguard
                    logger.debug(f"[{func_name}] Calculated fetch_limit is zero or negative. Breaking loop.")
                    break

            logger.debug(
                f"[{func_name}] Fetch Chunk #{request_count}: Since={current_since}, Limit={fetch_limit}, Params={params}"
            )

            candles_chunk: Optional[List[list]] = None
            last_fetch_error: Optional[Exception] = None

            # --- Internal Retry Loop for this Chunk ---
            max_chunk_retries = config.RETRY_COUNT  # Use config for retries
            for attempt in range(max_chunk_retries + 1):  # 0 to max_retries
                try:
                    # Ensure 'since' is int or None, not float
                    fetch_since = int(current_since) if current_since is not None else None
                    candles_chunk = exchange.fetch_ohlcv(
                        symbol, timeframe, since=fetch_since, limit=fetch_limit, params=params
                    )
                    last_fetch_error = None  # Reset error on success
                    logger.debug(f"[{func_name}] Chunk #{request_count} attempt {attempt + 1} succeeded.")
                    break  # Exit retry loop on success

                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded) as e:
                    last_fetch_error = e
                    if attempt < max_chunk_retries:
                        # Exponential backoff with jitter
                        retry_delay = config.RETRY_DELAY_SECONDS * (2**attempt) * (random.uniform(0.8, 1.2))
                        logger.warning(
                            f"{Fore.YELLOW}[{func_name}] API Error chunk #{request_count} (Try {attempt + 1}/{max_chunk_retries + 1}): {type(e).__name__}. Retrying in {retry_delay:.2f}s...{Style.RESET_ALL}"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"{Fore.RED}[{func_name}] API Error chunk #{request_count} failed after {max_chunk_retries + 1} attempts: {e}. Aborting chunk fetch.{Style.RESET_ALL}"
                        )

                except (
                    ccxt.ExchangeError
                ) as e:  # Includes BadSymbol, AuthenticationError etc. that shouldn't be retried usually
                    last_fetch_error = e
                    logger.error(
                        f"{Fore.RED}[{func_name}] Non-retryable ExchangeError on chunk #{request_count}: {e}. Aborting chunk fetch.{Style.RESET_ALL}"
                    )
                    break  # Don't retry logical exchange errors

                except Exception as e:  # Catch unexpected errors during fetch
                    last_fetch_error = e
                    logger.error(
                        f"{Fore.RED}[{func_name}] Unexpected error fetching chunk #{request_count}: {e}{Style.RESET_ALL}",
                        exc_info=True,
                    )
                    break  # Don't retry unknown errors
            # --- End Internal Retry Loop ---

            # If an error occurred and wasn't resolved by retries
            if last_fetch_error:
                logger.error(
                    f"{Fore.RED}[{func_name}] Failed to fetch chunk #{request_count} after {max_chunk_retries + 1} attempts. Last Error: {last_fetch_error}{Style.RESET_ALL}"
                )
                logger.warning(
                    f"[{func_name}] Returning potentially incomplete data ({len(all_candles)} candles) due to fetch failure in pagination loop."
                )
                break  # Exit the main pagination loop

            # Check if the exchange returned an empty list (no more data)
            if not candles_chunk:
                logger.info(
                    f"[{func_name}] No more candles returned by exchange (Chunk #{request_count}). End of data for this range."
                )
                break  # Exit main loop

            # --- Overlap Detection and Filtering ---
            # Only filter if we have previous candles AND the new chunk isn't empty
            if all_candles and candles_chunk:
                first_new_ts = candles_chunk[0][0]
                last_old_ts = all_candles[-1][0]

                if first_new_ts <= last_old_ts:
                    logger.warning(
                        f"{Fore.YELLOW}[{func_name}] Overlap detected chunk #{request_count} (New starts at {first_new_ts} <= Last fetched {last_old_ts}). Filtering overlap.{Style.RESET_ALL}"
                    )
                    # Keep only candles with timestamp strictly greater than the last one we have
                    original_chunk_len = len(candles_chunk)
                    candles_chunk = [c for c in candles_chunk if c[0] > last_old_ts]
                    filtered_count = original_chunk_len - len(candles_chunk)
                    if filtered_count > 0:
                        logger.debug(
                            f"[{func_name}] Removed {filtered_count} overlapping candles from chunk #{request_count}."
                        )

                    if not candles_chunk:
                        logger.info(
                            f"[{func_name}] Entire chunk #{request_count} was overlap or empty after filtering. Assuming end of useful data."
                        )
                        break  # Exit main loop

            # --- Add valid chunk data ---
            num_new_candles = len(candles_chunk)
            logger.debug(
                f"[{func_name}] Fetched {num_new_candles} new, non-overlapping candles (Chunk #{request_count}). Total collected: {len(all_candles) + num_new_candles}"
            )
            all_candles.extend(candles_chunk)

            # --- Check if exchange returned fewer than requested (usually end of data) ---
            # Only break if `since` was used (fetching forward). If fetching backwards (since=None),
            # fewer candles doesn't necessarily mean the end. However, fetch_ohlcv typically returns
            # data ending at the current time when since=None, so fewer than limit still implies end.
            if num_new_candles < fetch_limit:
                logger.info(
                    f"[{func_name}] Received fewer candles ({num_new_candles}) than requested ({fetch_limit}) in chunk #{request_count}. Assuming end of available data for this range."
                )
                break  # Exit main loop

            # --- Prepare 'since' for the next request ---
            # Timestamp of the last received candle + 1 timeframe interval
            # Ensure the timestamp is an integer
            current_since = int(candles_chunk[-1][0] + timeframe_ms)

            # --- Rate Limiting ---
            # Respect exchange rate limit (add a small buffer)
            rate_limit_ms = getattr(exchange, "rateLimit", 100)  # Default 100ms if not specified
            delay_seconds = (rate_limit_ms / 1000.0) * 1.1  # Add 10% buffer
            logger.trace(
                f"[{func_name}] Applying rate limit delay: {delay_seconds:.3f}s"
            )  # Use trace level if too verbose
            time.sleep(delay_seconds)

        # --- End Pagination Loop ---

        # Process the collected candles into the desired format (DataFrame or list)
        return _process_ohlcv_data(all_candles, func_name, symbol, timeframe, max_total_candles)

    except ccxt.BadSymbol as e:
        logger.error(f"{Fore.RED}[{func_name}] Invalid symbol '{symbol}' for OHLCV fetch: {e}{Style.RESET_ALL}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Exchange error during initial OHLCV setup ({symbol}, {timeframe}): {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        # Catch-all for unexpected errors during setup or loop logic (not chunk fetching)
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error during OHLCV pagination setup/loop: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


def _process_ohlcv_data(
    candle_list: List[list], parent_func_name: str, symbol: str, timeframe: str, max_candles: Optional[int] = None
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """
    Internal helper to process a list of OHLCV candles.

    If pandas is available, converts to a validated pandas DataFrame.
    Otherwise, returns the validated list of lists.

    Args:
        candle_list: The raw list of [ts, o, h, l, c, v] lists.
        parent_func_name: Name of the calling function for logging context.
        symbol: Market symbol.
        timeframe: Timeframe string.
        max_candles: Max number of candles expected (for trimming).

    Returns:
        - DataFrame if pandas is available and processing succeeds.
        - List of lists if pandas is not available.
        - Empty DataFrame/list if input is empty or processing fails critically.
        - None only in case of unexpected internal error during processing.
    """
    func_name = f"{parent_func_name}._process_ohlcv_data"
    cols = ["timestamp", "open", "high", "low", "close", "volume"]  # Including timestamp initially

    if not candle_list:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] No candles were collected for {symbol} ({timeframe}). Returning empty result.{Style.RESET_ALL}"
        )
        if PANDAS_AVAILABLE:
            empty_df = pd.DataFrame(columns=cols[1:]).astype({c: float for c in cols[1:]})  # OHLCV cols
            empty_df.index = pd.to_datetime([]).tz_localize("UTC")  # Empty UTC DatetimeIndex
            empty_df.index.name = "timestamp"
            return empty_df
        else:
            return []  # Return empty list

    logger.debug(f"[{func_name}] Processing {len(candle_list)} raw candles for {symbol} ({timeframe})...")

    # --- Data Validation (Applicable to both List and DataFrame processing) ---
    validated_list = []
    len(candle_list)
    seen_timestamps = set()
    invalid_entries = 0
    duplicate_timestamps = 0

    # Sort by timestamp first to handle duplicates consistently (keep first)
    candle_list.sort(key=lambda x: x[0])

    for i, candle in enumerate(candle_list):
        if not isinstance(candle, list) or len(candle) != 6:
            # logger.debug(f"[{func_name}] Skipping invalid entry (not list or wrong length) at index {i}: {candle}")
            invalid_entries += 1
            continue
        ts = candle[0]
        # Basic type check (can be enhanced)
        if not isinstance(ts, (int, float)) or ts <= 0:
            # logger.debug(f"[{func_name}] Skipping invalid entry (bad timestamp) at index {i}: {candle}")
            invalid_entries += 1
            continue
        # Check for duplicate timestamp
        if ts in seen_timestamps:
            # logger.debug(f"[{func_name}] Skipping duplicate timestamp entry at index {i}: {candle}")
            duplicate_timestamps += 1
            continue

        # Simple check for obviously bad OHLCV values (e.g., negative price/volume)
        # safe_decimal_conversion could be used here for more robustness if needed, but adds overhead
        if any(not isinstance(v, (int, float, str)) or (isinstance(v, (int, float)) and v < 0) for v in candle[1:]):
            # logger.debug(f"[{func_name}] Skipping invalid entry (negative or non-numeric OHLCV) at index {i}: {candle}")
            invalid_entries += 1
            continue

        validated_list.append(candle)
        seen_timestamps.add(ts)

    if invalid_entries > 0:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] Removed {invalid_entries} invalid entries during initial validation.{Style.RESET_ALL}"
        )
    if duplicate_timestamps > 0:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] Removed {duplicate_timestamps} duplicate timestamp entries.{Style.RESET_ALL}"
        )

    if not validated_list:
        logger.error(
            f"{Fore.RED}[{func_name}] No valid candles remaining after initial validation for {symbol} ({timeframe}). Returning empty result.{Style.RESET_ALL}"
        )
        if PANDAS_AVAILABLE:
            return _get_empty_ohlcv_df()  # Return standard empty DF
        else:
            return []

    # Trim if max_candles was specified and we collected more (due to overlap filtering logic maybe)
    if max_candles is not None and len(validated_list) > max_candles:
        logger.debug(
            f"[{func_name}] Trimming collected candles from {len(validated_list)} to the latest {max_candles}."
        )
        # Since list is sorted ascending, take the last N items
        validated_list = validated_list[-max_candles:]

    # --- Return List if Pandas is not available ---
    if not PANDAS_AVAILABLE:
        logger.info(
            f"{Fore.GREEN}[{func_name}] Successfully processed {len(validated_list)} valid candles for {symbol} ({timeframe}). Returning as list.{Style.RESET_ALL}"
        )
        return validated_list

    # --- Process with Pandas ---
    try:
        df = pd.DataFrame(validated_list, columns=cols)

        # Convert timestamp to DatetimeIndex (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)  # Remove rows where timestamp conversion failed
        if df.empty:
            logger.error(
                f"{Fore.RED}[{func_name}] DataFrame became empty after timestamp conversion/dropna for {symbol}.{Style.RESET_ALL}"
            )
            return _get_empty_ohlcv_df()

        df.set_index("timestamp", inplace=True)

        # Convert OHLCV columns to numeric (prefer float for typical analysis, use errors='coerce')
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for and handle NaNs introduced by coercion or from source
        nan_counts = df[ohlcv_cols].isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Found {total_nans} NaN values in OHLCV columns after conversion. Attempting forward fill... (Counts: {nan_counts[nan_counts > 0].to_dict()}){Style.RESET_ALL}"
            )
            df.ffill(inplace=True)  # Forward fill NaNs
            # Drop any remaining NaNs (e.g., at the very start if the first row had NaNs)
            initial_rows = len(df)
            df.dropna(subset=ohlcv_cols, inplace=True)
            rows_dropped = initial_rows - len(df)
            if rows_dropped > 0:
                logger.warning(
                    f"{Fore.YELLOW}[{func_name}] Dropped {rows_dropped} rows with persistent NaNs (likely at the beginning).{Style.RESET_ALL}"
                )

            # Final check if df became empty after NaN handling
            if df.empty:
                logger.error(
                    f"{Fore.RED}[{func_name}] DataFrame became empty after NaN handling for {symbol}.{Style.RESET_ALL}"
                )
                return _get_empty_ohlcv_df()

        # Ensure index is sorted (should be already, but good practice)
        if not df.index.is_monotonic_increasing:
            logger.debug(f"[{func_name}] Sorting DataFrame index.")
            df.sort_index(inplace=True)

        # Final trim (redundant if list was trimmed, but safe)
        if max_candles is not None and len(df) > max_candles:
            logger.debug(f"[{func_name}] Trimming DataFrame from {len(df)} to the last {max_candles} candles.")
            df = df.iloc[-max_candles:]

        if df.empty:  # Check again after potential trimming/dropna
            logger.error(
                f"{Fore.RED}[{func_name}] Processed DataFrame is empty after all cleaning steps for {symbol} ({timeframe}).{Style.RESET_ALL}"
            )
            return _get_empty_ohlcv_df()

        logger.info(
            f"{Fore.GREEN}[{func_name}] Successfully processed {len(df)} valid candles into DataFrame for {symbol} ({timeframe}).{Style.RESET_ALL}"
        )
        return df

    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Error processing OHLCV list into DataFrame: {e}{Style.RESET_ALL}", exc_info=True
        )
        # Return None or an empty DF? Let's return empty DF for consistency.
        return _get_empty_ohlcv_df()


def _get_empty_ohlcv_df() -> pd.DataFrame:
    """Helper to create a standardized empty OHLCV DataFrame."""
    if not PANDAS_AVAILABLE:
        return []  # Should not happen if called from pandas block
    cols = ["open", "high", "low", "close", "volume"]
    empty_df = pd.DataFrame(columns=cols).astype({c: float for c in cols})
    empty_df.index = pd.to_datetime([]).tz_localize("UTC")  # Empty UTC DatetimeIndex
    empty_df.index.name = "timestamp"
    return empty_df


# ============================================================================
# Function 11: Fetch Funding Rate
# ============================================================================
@retry_api_call(max_retries=3, initial_delay=1.0, caught_exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
def fetch_funding_rate(exchange: ccxt.bybit, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches the current funding rate details for a perpetual swap symbol on Bybit V5.

    Validates the symbol type and response data. Returns Decimals for rates/prices.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        config: The configuration object (currently unused here but kept for consistency).

    Returns:
        A dictionary containing funding rate details with numeric values as Decimals,
        or None if fetching fails, symbol is invalid, or data parsing fails.

    Raises:
        Re-raises ccxt.NetworkError, ccxt.ExchangeError for the retry decorator.
    """
    func_name = "fetch_funding_rate"
    logger.debug(f"[{func_name}] Fetching funding rate for {symbol}...")

    try:
        market = exchange.market(symbol)  # Can raise BadSymbol
        # Ensure it's a swap market
        if not market.get("swap", False):
            logger.error(
                f"{Fore.RED}[{func_name}] Symbol '{symbol}' is not a swap market. Cannot fetch funding rate.{Style.RESET_ALL}"
            )
            return None

        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            # V5 funding rates are typically only for linear/inverse perpetuals
            logger.error(
                f"{Fore.RED}[{func_name}] Invalid or unsupported category '{category}' determined for funding rate fetch ({symbol}). Expected 'linear' or 'inverse'.{Style.RESET_ALL}"
            )
            # If Bybit adds funding rates for options later, update this check.
            return None

        params = {"category": category}
        logger.debug(f"[{func_name}] Calling fetch_funding_rate with symbol='{symbol}', params={params}")

        # Fetch the funding rate using ccxt
        funding_rate_info = exchange.fetch_funding_rate(symbol, params=params)

        # --- Validate and Process Response ---
        if not funding_rate_info or not isinstance(funding_rate_info, dict):
            logger.error(
                f"{Fore.RED}[{func_name}] Received invalid or empty funding rate response for {symbol}.{Style.RESET_ALL}"
            )
            return None

        processed_fr: Dict[str, Any] = {}
        errors: List[str] = []

        # Process and validate essential fields
        processed_fr["symbol"] = funding_rate_info.get("symbol", symbol)  # Use original symbol as fallback
        processed_fr["fundingTimestamp"] = funding_rate_info.get("fundingTimestamp")  # Milliseconds UTC
        processed_fr["fundingDatetime"] = funding_rate_info.get("fundingDatetime")  # ISO8601 String

        fr_val = funding_rate_info.get("fundingRate")
        processed_fr["fundingRate"] = safe_decimal_conversion(fr_val)
        if processed_fr["fundingRate"] is None:
            errors.append(f"Could not parse 'fundingRate' ({fr_val})")

        mp_val = funding_rate_info.get("markPrice")
        processed_fr["markPrice"] = safe_decimal_conversion(mp_val)
        if processed_fr["markPrice"] is None:
            # Mark price might be temporarily unavailable? Log warning.
            logger.warning(f"[{func_name}] Could not parse 'markPrice' ({mp_val}) for {symbol}.")

        ip_val = funding_rate_info.get("indexPrice")
        processed_fr["indexPrice"] = safe_decimal_conversion(ip_val)
        if processed_fr["indexPrice"] is None:
            logger.warning(f"[{func_name}] Could not parse 'indexPrice' ({ip_val}) for {symbol}.")

        # Handle next funding time (CCXT might return 'nextFundingTimestamp' or similar)
        next_ts_key = "nextFundingTimestamp"  # Common key from ccxt structure
        next_ts_val = funding_rate_info.get(next_ts_key) or funding_rate_info.get(
            "nextFundingTime"
        )  # Check alternative if needed
        processed_fr["nextFundingTime"] = next_ts_val  # Store raw timestamp (milliseconds)
        processed_fr["nextFundingDatetime"] = None  # Initialize

        if next_ts_val and isinstance(next_ts_val, (int, float)) and next_ts_val > 0:
            try:
                # Use pandas for robust datetime conversion and formatting if available
                if PANDAS_AVAILABLE:
                    next_dt = pd.to_datetime(next_ts_val, unit="ms", utc=True)
                    processed_fr["nextFundingDatetime"] = next_dt.strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    )  # e.g., "2023-10-27 16:00:00 UTC"
                else:
                    # Basic formatting without pandas (less robust)
                    import datetime

                    next_dt_basic = datetime.datetime.fromtimestamp(next_ts_val / 1000, tz=datetime.timezone.utc)
                    processed_fr["nextFundingDatetime"] = next_dt_basic.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception as dt_err:
                logger.warning(
                    f"[{func_name}] Could not format next funding datetime from timestamp {next_ts_val}: {dt_err}"
                )
                processed_fr["nextFundingDatetime"] = "Error formatting"
        elif next_ts_val:
            logger.warning(
                f"[{func_name}] Found next funding time key ('{next_ts_key}'), but value is invalid: {next_ts_val}"
            )

        # Include raw info
        processed_fr["info"] = funding_rate_info.get("info", {})

        # --- Final Check and Logging ---
        if errors:
            # If fundingRate is missing, treat as failure
            logger.error(
                f"{Fore.RED}[{func_name}] Failed to process funding rate response for {symbol}. Errors: {'; '.join(errors)}. Raw: {funding_rate_info}{Style.RESET_ALL}"
            )
            return None

        rate = processed_fr.get("fundingRate")
        next_dt_str = processed_fr.get("nextFundingDatetime", "N/A")
        rate_str = f"{rate:.6%}" if rate is not None else "N/A"  # Format as percentage
        logger.info(
            f"{Fore.GREEN}[{func_name}] Funding Rate Fetched {symbol}: Rate = {rate_str}, Next = {next_dt_str}{Style.RESET_ALL}"
        )

        return processed_fr

    except ccxt.BadSymbol as e:
        logger.error(f"{Fore.RED}[{func_name}] Invalid symbol '{symbol}' for funding rate fetch: {e}{Style.RESET_ALL}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] API Error fetching funding rate for {symbol}: {e}. Retrying if possible...{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching funding rate for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# ============================================================================
# Function 13: Fetch L2 Order Book (Validated)
# ============================================================================
@retry_api_call(max_retries=2, initial_delay=0.5, caught_exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ValueError))
def fetch_l2_order_book_validated(
    exchange: ccxt.bybit, symbol: str, limit: int, config: Config
) -> Optional[Dict[str, List[Tuple[Decimal, Decimal]]]]:
    """
    Fetches the Level 2 order book for a symbol using Bybit V5 fetchOrderBook.

    Validates the structure, price/amount data, and checks for crossed books.
    Returns bids and asks as lists of [price, amount] tuples using Decimals,
    sorted correctly (bids descending, asks ascending by price).

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        limit: Desired depth of the order book (number of bids/asks). Will be clamped by category limits.
        config: Configuration object (used for default limit if needed).

    Returns:
        A dictionary {'bids': [...], 'asks': [...]} with validated and sorted
        price/amount tuples as Decimals, or None if fetching or validation fails critically.

    Raises:
        Re-raises ccxt.NetworkError, ccxt.ExchangeError, ValueError for the retry decorator.
    """
    func_name = "fetch_l2_order_book_validated"
    logger.debug(f"[{func_name}] Fetching L2 Order Book for {symbol} (Requested Limit: {limit})...")

    if not hasattr(exchange, "fetch_order_book") or not exchange.has.get("fetchOrderBook"):
        logger.error(
            f"{Fore.RED}[{func_name}] Exchange '{exchange.id}' does not support fetchOrderBook.{Style.RESET_ALL}"
        )
        return None

    try:
        market = exchange.market(symbol)  # Can raise BadSymbol
        category = _get_v5_category(market)

        params = {}
        # Determine V5 specific limit constraints based on category
        # From Bybit Docs (approximate, check current docs): Spot: 200, Linear/Inverse: 500, Option: 25/50?
        # CCXT might handle clamping, but we can pre-clamp for clarity.
        max_limit_map = {"spot": 200, "linear": 500, "inverse": 500, "option": 50}
        default_max_limit = 500  # A reasonable default if category unknown

        if category:
            params = {"category": category}
            max_limit = max_limit_map.get(category, default_max_limit)
        else:
            logger.warning(
                f"[{func_name}] Cannot determine category for {symbol}. Fetching OB without category param, max limit assumed {default_max_limit}."
            )
            max_limit = default_max_limit

        # Validate and clamp the requested limit
        effective_limit = limit
        if not isinstance(limit, int) or limit <= 0:
            default_limit = getattr(config, "ORDER_BOOK_FETCH_LIMIT", 25)  # Use config default
            logger.warning(f"[{func_name}] Invalid limit requested ({limit}). Using default: {default_limit}.")
            effective_limit = default_limit

        if effective_limit > max_limit:
            logger.warning(
                f"[{func_name}] Requested limit {effective_limit} exceeds max {max_limit} for category '{category or 'Unknown'}'. Clamping to {max_limit}."
            )
            effective_limit = max_limit

        # --- Fetch Order Book ---
        logger.debug(
            f"[{func_name}] Calling fetchOrderBook with symbol='{symbol}', limit={effective_limit}, params={params}"
        )
        order_book = exchange.fetch_order_book(symbol, limit=effective_limit, params=params)

        # --- Basic Structure Validation ---
        if not isinstance(order_book, dict) or "bids" not in order_book or "asks" not in order_book:
            logger.error(
                f"{Fore.RED}[{func_name}] Invalid order book structure received: Keys missing or not a dict.{Style.RESET_ALL}"
            )
            raise ValueError("Invalid order book structure received (missing bids/asks or not dict).")
        if not isinstance(order_book["bids"], list) or not isinstance(order_book["asks"], list):
            logger.error(
                f"{Fore.RED}[{func_name}] Bids or asks are not lists: bids_type={type(order_book['bids'])}, asks_type={type(order_book['asks'])}{Style.RESET_ALL}"
            )
            raise ValueError("Order book 'bids' or 'asks' are not lists.")

        # --- Process and Validate Entries ---
        validated_bids: List[Tuple[Decimal, Decimal]] = []
        validated_asks: List[Tuple[Decimal, Decimal]] = []
        conversion_errors = 0

        # Process bids (expecting [price, amount] format from CCXT)
        for i, entry in enumerate(order_book.get("bids", [])):
            if not isinstance(entry, list) or len(entry) != 2:
                conversion_errors += 1
                logger.debug(f"[{func_name}] Skipping invalid bid entry format at index {i}: {entry}")
                continue
            price = safe_decimal_conversion(entry[0])
            amount = safe_decimal_conversion(entry[1])
            # Validate: Price must be positive, Amount must be non-negative
            if price is None or amount is None or price <= Decimal("0") or amount < Decimal("0"):
                conversion_errors += 1
                logger.debug(
                    f"[{func_name}] Skipping invalid bid values at index {i}: Price={entry[0]}, Amount={entry[1]}"
                )
                continue
            validated_bids.append((price, amount))

        # Process asks
        for i, entry in enumerate(order_book.get("asks", [])):
            if not isinstance(entry, list) or len(entry) != 2:
                conversion_errors += 1
                logger.debug(f"[{func_name}] Skipping invalid ask entry format at index {i}: {entry}")
                continue
            price = safe_decimal_conversion(entry[0])
            amount = safe_decimal_conversion(entry[1])
            # Validate: Price must be positive, Amount must be non-negative
            if price is None or amount is None or price <= Decimal("0") or amount < Decimal("0"):
                conversion_errors += 1
                logger.debug(
                    f"[{func_name}] Skipping invalid ask values at index {i}: Price={entry[0]}, Amount={entry[1]}"
                )
                continue
            validated_asks.append((price, amount))

        if conversion_errors > 0:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Skipped {conversion_errors} invalid price/amount entries in order book for {symbol}.{Style.RESET_ALL}"
            )

        # --- Sorting Validation (CCXT usually guarantees this) ---
        # Bids should be descending by price, Asks ascending by price
        if validated_bids and any(
            validated_bids[i][0] < validated_bids[i + 1][0] for i in range(len(validated_bids) - 1)
        ):
            logger.warning(f"{Fore.YELLOW}[{func_name}] Bids order incorrect. Re-sorting descending.{Style.RESET_ALL}")
            validated_bids.sort(key=lambda x: x[0], reverse=True)
        if validated_asks and any(
            validated_asks[i][0] > validated_asks[i + 1][0] for i in range(len(validated_asks) - 1)
        ):
            logger.warning(f"{Fore.YELLOW}[{func_name}] Asks order incorrect. Re-sorting ascending.{Style.RESET_ALL}")
            validated_asks.sort(key=lambda x: x[0])

        # --- Crossed Book Validation ---
        if validated_bids and validated_asks:
            best_bid = validated_bids[0][0]  # Highest bid price after sorting
            best_ask = validated_asks[0][0]  # Lowest ask price after sorting
            if best_bid >= best_ask:
                # This is a serious data quality issue
                logger.error(
                    f"{Back.RED}[{func_name}] CRITICAL: Order book is CROSSED for {symbol}! Best Bid ({best_bid}) >= Best Ask ({best_ask}). Data is unreliable.{Style.RESET_ALL}"
                )
                # Depending on strategy, either raise ValueError, return None, or return the flawed data
                # Raising ValueError allows the retry decorator to potentially catch transient crossed books
                raise ValueError(f"Order book crossed: Best Bid {best_bid} >= Best Ask {best_ask}")
                # return None # Alternative: Fail silently without retry

        # Check if bids/asks became empty after validation
        if not validated_bids and not validated_asks and (order_book["bids"] or order_book["asks"]):
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] All bid/ask entries failed validation for {symbol}. Returning empty book.{Style.RESET_ALL}"
            )
            # Fall through to return empty lists

        logger.info(
            f"{Fore.GREEN}[{func_name}] Successfully fetched and validated L2 Order Book for {symbol}. Bids: {len(validated_bids)}, Asks: {len(validated_asks)}{Style.RESET_ALL}"
        )
        return {"bids": validated_bids, "asks": validated_asks}

    except ccxt.BadSymbol as e:
        logger.error(f"{Fore.RED}[{func_name}] Invalid symbol '{symbol}' for order book fetch: {e}{Style.RESET_ALL}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
        # Log as warning/error depending on type, raise for retry decorator
        log_level = logging.ERROR if isinstance(e, ValueError) else logging.WARNING
        logger.log(
            log_level,
            f"{Fore.YELLOW if log_level == logging.WARNING else Fore.RED}[{func_name}] API/Validation Error fetching L2 OB for {symbol}: {e}. Retrying if possible...{Style.RESET_ALL}",
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


# ============================================================================
# Function 17: Fetch Ticker (Validated)
# ============================================================================
@retry_api_call(max_retries=2, initial_delay=0.5, caught_exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ValueError))
def fetch_ticker_validated(
    exchange: ccxt.bybit, symbol: str, config: Config, max_age_seconds: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fetches the ticker for a symbol from Bybit V5, validates its freshness and key values.

    Checks timestamp, essential prices (last, bid, ask), calculates spread,
    and validates volume fields. Returns a dictionary with Decimal values where appropriate.

    Args:
        exchange: Initialized ccxt.bybit exchange instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        config: Configuration object (currently unused but kept for consistency).
        max_age_seconds: Maximum acceptable age of the ticker data in seconds.

    Returns:
        A dictionary containing validated ticker information with numeric values
        as Decimals where appropriate, or None if fetching or validation fails.

    Raises:
        Re-raises ccxt.NetworkError, ccxt.ExchangeError, ValueError for the retry decorator.
        ValueError is raised for critical validation failures (stale, bad prices, crossed spread).
    """
    func_name = "fetch_ticker_validated"
    logger.debug(f"[{func_name}] Fetching and validating ticker for {symbol} (Max Age: {max_age_seconds}s)...")

    try:
        market = exchange.market(symbol)  # Can raise BadSymbol
        category = _get_v5_category(market)

        params = {}
        if category:
            params = {"category": category}
        else:
            logger.warning(
                f"[{func_name}] Cannot determine category for {symbol}. Fetching ticker without category param."
            )

        # --- Fetch Ticker ---
        logger.debug(f"[{func_name}] Calling fetch_ticker with symbol='{symbol}', params={params}")
        ticker = exchange.fetch_ticker(symbol, params=params)

        # --- Basic Structure Validation ---
        if not ticker or not isinstance(ticker, dict):
            logger.error(
                f"{Fore.RED}[{func_name}] Received invalid or empty ticker response for {symbol}.{Style.RESET_ALL}"
            )
            raise ValueError("Invalid or empty ticker response received.")

        # --- Timestamp Validation (Freshness) ---
        timestamp_ms = ticker.get("timestamp")
        current_time_ms = time.time() * 1000

        if timestamp_ms is None or not isinstance(timestamp_ms, (int, float)) or timestamp_ms <= 0:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Ticker for {symbol} has missing or invalid timestamp: {timestamp_ms}. Proceeding cautiously.{Style.RESET_ALL}"
            )
            # Allow proceeding but data might be questionable. A stricter approach could raise ValueError here.
            age_seconds = float("inf")  # Treat as stale if timestamp is bad
        else:
            age_seconds = (current_time_ms - timestamp_ms) / 1000.0

            if age_seconds > max_age_seconds:
                error_msg = f"Ticker for {symbol} is stale. Age: {age_seconds:.1f}s > Max: {max_age_seconds}s."
                logger.error(f"{Fore.RED}[{func_name}] {error_msg}{Style.RESET_ALL}")
                raise ValueError(error_msg)  # Raise for retry or failure

            # Check for clocks significantly out of sync (e.g., ticker from the future)
            # Allow a small grace period (e.g., 5 seconds) for minor sync differences
            if age_seconds < -5.0:
                error_msg = f"Ticker timestamp {timestamp_ms} seems to be from the future (Current: {current_time_ms:.0f}, Diff: {age_seconds:.1f}s). Check system clock sync."
                logger.error(f"{Fore.RED}[{func_name}] {error_msg}{Style.RESET_ALL}")
                raise ValueError(error_msg)  # Clock sync issue is critical

        # --- Price and Volume Validation ---
        validated_ticker: Dict[str, Any] = {"symbol": ticker.get("symbol", symbol)}  # Start building validated dict
        errors: List[str] = []

        # Essential prices
        last_price = safe_decimal_conversion(ticker.get("last"))
        bid_price = safe_decimal_conversion(ticker.get("bid"))
        ask_price = safe_decimal_conversion(ticker.get("ask"))

        if last_price is None or last_price <= Decimal("0"):
            errors.append(f"Invalid or missing 'last' price: {ticker.get('last')}")
        if bid_price is None or bid_price <= Decimal("0"):
            # Bid/Ask might be None in illiquid markets, log warning but don't necessarily fail validation unless needed
            logger.warning(f"[{func_name}] Ticker for {symbol} has invalid or missing 'bid' price: {ticker.get('bid')}")
            # errors.append(f"Invalid or missing 'bid' price: {ticker.get('bid')}") # Uncomment if bid is essential
        if ask_price is None or ask_price <= Decimal("0"):
            logger.warning(f"[{func_name}] Ticker for {symbol} has invalid or missing 'ask' price: {ticker.get('ask')}")
            # errors.append(f"Invalid or missing 'ask' price: {ticker.get('ask')}") # Uncomment if ask is essential

        validated_ticker.update(
            {
                "timestamp": timestamp_ms,
                "datetime": ticker.get("datetime"),  # ISO8601 string
                "last": last_price,
                "bid": bid_price,
                "ask": ask_price,
            }
        )

        # Spread Calculation and Validation
        spread: Optional[Decimal] = None
        spread_pct: Optional[Decimal] = None
        if bid_price and ask_price and bid_price > 0 and ask_price > 0:
            if bid_price >= ask_price:
                # This is a critical error indicating bad data
                error_msg = f"Ticker spread is invalid (crossed): Bid ({bid_price}) >= Ask ({ask_price})"
                errors.append(error_msg)
                # No need to calculate spread if crossed
            else:
                spread = ask_price - bid_price
                try:
                    # Use ask price for percentage denominator for stability? Or mid-price? Bid is common.
                    spread_pct = (spread / ask_price) * Decimal("100") if ask_price > 0 else Decimal("inf")
                except InvalidOperation:  # Catch potential issues if ask is extremely small
                    spread_pct = Decimal("inf")
        validated_ticker["spread"] = spread
        validated_ticker["spread_pct"] = spread_pct

        # Other numeric fields (validate non-negative where applicable)
        numeric_fields = {
            "bidVolume": "bidVolume",
            "askVolume": "askVolume",
            "baseVolume": "baseVolume",
            "quoteVolume": "quoteVolume",
            "high": "high",
            "low": "low",
            "open": "open",
            "change": "change",
            "percentage": "percentage",
            "average": "average",
        }
        for key, source_key in numeric_fields.items():
            val = safe_decimal_conversion(ticker.get(source_key))
            # Volumes should be non-negative
            if "Volume" in key and val is not None and val < Decimal("0"):
                logger.warning(f"[{func_name}] Ticker for {symbol} has negative {key}: {val}. Setting to 0.")
                val = Decimal("0.0")
            # Prices (high, low, open, average) should ideally be positive if not None
            elif key in ["high", "low", "open", "average"] and val is not None and val <= Decimal("0"):
                logger.warning(f"[{func_name}] Ticker for {symbol} has non-positive {key}: {val}.")
                # Decide if this is an error or just a warning
            validated_ticker[key] = val

        # CCXT convention: 'close' is often the same as 'last' for tickers
        validated_ticker["close"] = last_price

        # Raw info
        validated_ticker["info"] = ticker.get("info", {})

        # --- Final Validation Check ---
        if errors:
            error_summary = f"Ticker validation failed for {symbol}: {'; '.join(errors)}"
            logger.error(f"{Fore.RED}[{func_name}] {error_summary}{Style.RESET_ALL}")
            raise ValueError(error_summary)  # Raise for retry/failure

        # --- Logging Success ---
        spread_str = f"{spread_pct:.4f}%" if spread_pct is not None and spread_pct != Decimal("inf") else "N/A"
        age_str = f"{age_seconds:.1f}s" if age_seconds != float("inf") else "Timestamp Invalid"
        logger.info(
            f"{Fore.GREEN}[{func_name}] Ticker validation OK for {symbol}: Last={format_price(exchange, symbol, last_price)}, "
            f"Bid={format_price(exchange, symbol, bid_price)}, Ask={format_price(exchange, symbol, ask_price)}, "
            f"Spread={spread_str} (Age: {age_str}){Style.RESET_ALL}"
        )
        return validated_ticker

    except ccxt.BadSymbol as e:
        logger.error(f"{Fore.RED}[{func_name}] Invalid symbol '{symbol}' for ticker fetch: {e}{Style.RESET_ALL}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
        log_level = logging.ERROR if isinstance(e, ValueError) else logging.WARNING
        logger.log(
            log_level,
            f"{Fore.YELLOW if log_level == logging.WARNING else Fore.RED}[{func_name}] Failed to fetch or validate ticker for {symbol}: {e}. Retrying if possible...{Style.RESET_ALL}",
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching/validating ticker for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# ============================================================================
# Function 21: Fetch Recent Trades
# ============================================================================
@retry_api_call(max_retries=2, initial_delay=0.5, caught_exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
def fetch_recent_trades(
    exchange: ccxt.bybit, symbol: str, config: Config, limit: int = 100, min_size_filter: Optional[Decimal] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches recent public trades for a symbol from Bybit V5, validates data,
    and returns a list of trade dictionaries with Decimal values, sorted recent first.

    Optionally filters trades smaller than `min_size_filter` (in base currency amount).

    Args:
        exchange: Initialized ccxt.bybit exchange instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        config: Configuration object (currently unused but kept for consistency).
        limit: Maximum number of trades to fetch (subject to exchange limits, typically 1000 for Bybit V5).
        min_size_filter: Optional minimum trade amount (base currency) to include.

    Returns:
        A list of validated trade dictionaries, sorted by timestamp descending,
        or None if fetching fails critically. Returns an empty list if no trades are found
        or if all trades are filtered out.

    Raises:
        Re-raises ccxt.NetworkError, ccxt.ExchangeError for the retry decorator.
    """
    func_name = "fetch_recent_trades"
    filter_log = (
        f"(MinSize: {format_amount(exchange, symbol, min_size_filter) if min_size_filter else 'None'})"
        if min_size_filter
        else ""
    )
    logger.debug(f"[{func_name}] Fetching recent trades for {symbol} (Requested Limit: {limit}) {filter_log}...")

    # Validate and clamp limit
    # Bybit V5 limit for public trades (market execution history) is typically 1000
    max_trade_limit = 1000
    effective_limit = limit
    if not isinstance(limit, int) or limit <= 0:
        logger.warning(f"[{func_name}] Invalid limit requested ({limit}). Using default: 100.")
        effective_limit = 100
    elif limit > max_trade_limit:
        logger.warning(
            f"[{func_name}] Requested limit {limit} exceeds max {max_trade_limit}. Clamping to {max_trade_limit}."
        )
        effective_limit = max_trade_limit

    try:
        market = exchange.market(symbol)  # Can raise BadSymbol
        category = _get_v5_category(market)

        params = {}
        if category:
            params = {"category": category}
            # Bybit V5 specific params if needed:
            # params['optionType'] = 'Trade' # Might filter noise if applicable? Check docs.
            # params['baseCoin'] = market['base'] # For Spot v5? Check docs.
        else:
            logger.warning(
                f"[{func_name}] Cannot determine category for {symbol}. Fetching trades without category param."
            )

        # --- Fetch Trades ---
        logger.debug(
            f"[{func_name}] Calling fetch_trades with symbol='{symbol}', limit={effective_limit}, params={params}"
        )
        trades_raw = exchange.fetch_trades(symbol, limit=effective_limit, params=params)

        if trades_raw is None:
            # Distinguish between API error (caught below) and valid empty response
            logger.error(
                f"{Fore.RED}[{func_name}] fetch_trades returned None, indicating a potential issue with the request or CCXT implementation.{Style.RESET_ALL}"
            )
            return None  # Treat None response as failure
        if not trades_raw:  # Empty list is valid, means no trades found
            logger.info(f"[{func_name}] No recent trades found for {symbol} with current parameters.")
            return []

        # --- Process and Validate Trades ---
        processed_trades: List[Dict] = []
        conversion_errors = 0
        filtered_out_count = 0
        required_keys = [
            "id",
            "timestamp",
            "datetime",
            "symbol",
            "side",
            "price",
            "amount",
        ]  # Essential keys from CCXT structure

        for i, trade in enumerate(trades_raw):
            if not isinstance(trade, dict):
                conversion_errors += 1
                logger.debug(f"[{func_name}] Skipping non-dict trade entry at index {i}: {trade}")
                continue

            # Check for essential keys before processing
            if not all(key in trade and trade[key] is not None for key in required_keys):
                missing = [key for key in required_keys if key not in trade or trade[key] is None]
                conversion_errors += 1
                logger.debug(
                    f"[{func_name}] Skipping trade with missing essential keys ({missing}) at index {i}: {trade}"
                )
                continue

            try:
                # Validate and convert core fields
                trade_id = str(trade["id"])  # Ensure ID is string
                timestamp = int(trade["timestamp"])  # Ensure timestamp is int
                side = str(trade["side"]).lower()  # Normalize side to lowercase ('buy' or 'sell')
                price = safe_decimal_conversion(trade["price"])
                amount = safe_decimal_conversion(trade["amount"])

                # Check numeric values are valid
                if price is None or amount is None or price <= Decimal("0") or amount <= Decimal("0"):
                    conversion_errors += 1
                    logger.debug(
                        f"[{func_name}] Skipping trade with invalid price/amount at index {i}: Price={trade['price']}, Amount={trade['amount']}"
                    )
                    continue
                if side not in ["buy", "sell"]:
                    conversion_errors += 1
                    logger.debug(f"[{func_name}] Skipping trade with invalid side '{trade['side']}' at index {i}")
                    continue

                # Apply size filter if specified
                if min_size_filter is not None and amount < min_size_filter:
                    filtered_out_count += 1
                    continue

                # Calculate or validate cost
                cost = safe_decimal_conversion(trade.get("cost"))
                calculated_cost = price * amount
                # Use calculated cost if original 'cost' is missing or significantly different
                # Allow a small relative tolerance (e.g., 0.1%) for rounding differences
                if cost is None or abs(cost - calculated_cost) > (calculated_cost * Decimal("0.001")):
                    if cost is not None:
                        logger.trace(
                            f"[{func_name}] Trade cost mismatch (Provided: {cost}, Calculated: {calculated_cost}) for trade {trade_id}. Using calculated cost."
                        )
                    cost = calculated_cost

                processed_trades.append(
                    {
                        "id": trade_id,
                        "timestamp": timestamp,  # Milliseconds UTC
                        "datetime": trade.get("datetime"),  # ISO8601 string
                        "symbol": trade.get("symbol", symbol),  # Use original symbol as fallback
                        "side": side,
                        "price": price,
                        "amount": amount,
                        "cost": cost,
                        "takerOrMaker": trade.get("takerOrMaker"),  # 'taker' or 'maker' or None
                        "info": trade.get("info", {}),  # Raw exchange response for this trade
                    }
                )
            except (ValueError, TypeError, KeyError, InvalidOperation) as proc_err:
                conversion_errors += 1
                logger.warning(
                    f"{Fore.YELLOW}[{func_name}] Error processing individual trade at index {i}: {proc_err}. Data: {trade}{Style.RESET_ALL}"
                )
            except Exception as E:  # Catch any other unexpected error during processing
                conversion_errors += 1
                logger.error(
                    f"{Fore.RED}[{func_name}] Unexpected error processing trade at index {i}: {E}. Data: {trade}{Style.RESET_ALL}",
                    exc_info=True,
                )

        if conversion_errors > 0:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Skipped {conversion_errors} trades due to processing/validation errors for {symbol}.{Style.RESET_ALL}"
            )
        if filtered_out_count > 0:
            logger.info(
                f"[{func_name}] Filtered out {filtered_out_count} trades smaller than {min_size_filter} for {symbol}."
            )

        # Sort trades by timestamp descending (most recent first) - CCXT usually returns this way, but good practice to ensure
        if processed_trades:
            processed_trades.sort(key=lambda x: x["timestamp"], reverse=True)

        logger.info(
            f"{Fore.GREEN}[{func_name}] Fetched and processed {len(processed_trades)} valid trades for {symbol} {filter_log} (Limit: {effective_limit}).{Style.RESET_ALL}"
        )
        return processed_trades

    except ccxt.BadSymbol as e:
        logger.error(f"{Fore.RED}[{func_name}] Invalid symbol '{symbol}' for trades fetch: {e}{Style.RESET_ALL}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] API Error fetching trades for {symbol}: {e}. Retrying if possible...{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching trades for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- END OF FILE market_data.py ---
# ---------------------------------------------------------------------------
