# File: exchange_api.py
"""
Exchange API module for interacting with Bybit cryptocurrency exchange
using CCXT async support within a class structure.

This module provides functions for:
- Connecting to the Bybit API (async)
- Fetching market data (ticker, klines, orderbook) with retry and validation (async)
- Managing orders (create, cancel, query) with parameter handling (async)
- Managing positions (open, close, query) with standardization (async)
- Retrieving account information (balance, margin) (async)
- Setting leverage (async)
- Setting position protection (Stop Loss, Take Profit, Trailing Stop) (async, Bybit V5 specific)
"""

import os
import time
import hmac
import hashlib
import json
import logging
import urllib.parse
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio # Explicitly import asyncio for async operations and sleep
import importlib.metadata # For getting package version

# Use async support version of ccxt
import ccxt.async_support as ccxt_async # Renamed to avoid conflict if synchronous ccxt is used elsewhere
import pandas as pd # For DataFrame in fetch_klines_ccxt
from dotenv import load_dotenv # For loading environment variables

# Load environment variables from a .env file
load_dotenv()

# Import constants and utility functions from your utils module
# Ensure your utils.py has NEON_GREEN, NEON_RED, NEON_YELLOW, RESET_ALL_STYLE,
# MAX_API_RETRIES, RETRY_DELAY_SECONDS, get_min_tick_size, get_price_precision
# and potentially format_signal if used elsewhere.
from utils import (
    MAX_API_RETRIES,
    NEON_GREEN, NEON_RED, NEON_YELLOW, RESET_ALL_STYLE,
    RETRY_DELAY_SECONDS,
    get_min_tick_size,
    get_price_precision,
    # format_signal, # Uncomment if format_signal is needed in this module
)

# Module-level logger for general messages or before class instantiation
module_logger = logging.getLogger(__name__)

class BybitAPI:
    """
    Asynchronous Bybit API client for cryptocurrency trading using CCXT.

    This class encapsulates the CCXT exchange instance and provides asynchronous
    methods for interacting with the Bybit API, including enhanced error handling,
    retry logic, data validation, and specific parameter handling for Bybit V5.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Bybit API client.

        Args:
            config: The application configuration dictionary, expected to contain
                    'api_key', 'api_secret', 'use_sandbox', 'exchange_options', etc.
            logger: The logger instance to use for this API client instance.
        """
        self.logger = logger # Use the logger provided by the caller

        # Load API credentials from config or environment variables
        # Prioritize config, then environment variables
        api_key = config.get("api_key") or os.environ.get("BYBIT_API_KEY")
        api_secret = config.get("api_secret") or os.environ.get("BYBIT_API_SECRET")

        if not api_key or not api_secret:
            self.logger.error(f"{NEON_RED}BYBIT_API_KEY and BYBIT_API_SECRET must be provided in config or environment variables.{RESET_ALL_STYLE}")
            raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set.")

        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = config.get("use_sandbox", False) # Use 'use_sandbox' from config
        self.exchange_id = config.get("exchange_id", "bybit").lower()

        if self.exchange_id != 'bybit':
             self.logger.warning(f"{NEON_YELLOW}BybitAPI class is designed for 'bybit', but config specifies '{self.exchange_id}'. Behavior may be unexpected.{RESET_ALL_STYLE}")

        # Configure CCXT async client
        if not hasattr(ccxt_async, self.exchange_id):
            self.logger.error(f"{NEON_RED}Exchange ID '{self.exchange_id}' not found in CCXT async library.{RESET_ALL_STYLE}")
            raise ValueError(f"Exchange ID '{self.exchange_id}' not supported by CCXT async.")

        exchange_class = getattr(ccxt_async, self.exchange_id)

        # Retrieve exchange-specific options from config, merge with essential defaults
        # This allows main.py to control defaultType, timeouts, etc.
        exchange_options = {
             'apiKey': self.api_key,
             'secret': self.api_secret,
             'enableRateLimit': True, # Enable CCXT's built-in rate limiter
             # Start with options from config, then add/override defaults
             'options': config.get('exchange_options', {}).get('options', {}).copy(),
             # Merge any other top-level options from config if needed, e.g., 'password'
        }

        # Ensure defaultType is set in options, fallback to 'unified' for Bybit, 'linear' otherwise
        # Check if defaultType is already explicitly set in config options
        if 'defaultType' not in exchange_options['options']:
             exchange_options['options']['defaultType'] = 'unified' if self.exchange_id == 'bybit' else 'linear'
             self.logger.debug(f"Setting defaultType in options to: {exchange_options['options']['defaultType']}")
        else:
             self.logger.debug(f"Using defaultType from config options: {exchange_options['options']['defaultType']}")


        # Add Bybit specific overrides/defaults if not already in config options
        if self.exchange_id == 'bybit':
            # Bybit: Market orders do not require a price parameter
            if 'createOrderRequiresPrice' not in exchange_options['options']:
                 exchange_options['options']['createOrderRequiresPrice'] = False
                 self.logger.debug("Setting createOrderRequiresPrice to False for Bybit.")
            # Bybit V5 often needs 'recvWindow' param. Default to a safe value if not set.
            # This is often passed in params for specific calls, but can be a default option.
            # Let's add it to default params for relevant calls instead of here.
            # if 'recvWindow' not in exchange_options['options']:
            #      exchange_options['options']['recvWindow'] = 5000 # milliseconds
            #      self.logger.debug("Setting recvWindow default option to 5000 for Bybit.")

        # Instantiate the exchange class
        self.exchange: ccxt_async.Exchange = exchange_class(exchange_options)

        # Configure sandbox mode if enabled in config
        if self.testnet:
            self.logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet) for {self.exchange.id}{RESET_ALL_STYLE}")
            # CCXT standard way to enable sandbox
            if hasattr(self.exchange, 'set_sandbox_mode') and callable(self.exchange.set_sandbox_mode):
                try:
                    # set_sandbox_mode is an async method
                    # Since __init__ is sync, we need to run this in a temporary event loop
                    # This is generally discouraged in __init__, consider moving exchange setup
                    # to an async connect() method called after instantiation.
                    # For now, using asyncio.run for simplicity, but be aware of potential issues.
                    asyncio.run(self.exchange.set_sandbox_mode(True))
                    self.logger.info(f"Sandbox mode enabled for {self.exchange.id} via set_sandbox_mode(True).")
                except Exception as sandbox_err:
                    self.logger.warning(
                        f"Error calling set_sandbox_mode(True) for {self.exchange.id}: {sandbox_err}. "
                        f"Attempting manual URL override if known for Bybit."
                    )
                    # Fallback: Manual URL override for Bybit if set_sandbox_mode fails
                    if self.exchange.id == 'bybit':
                         testnet_url = self.exchange.urls.get('test', 'https://api-testnet.bybit.com')
                         self.exchange.urls['api'] = testnet_url
                         self.logger.info(f"Manual Bybit testnet URL set: {testnet_url}")
                    else:
                         self.logger.warning(f"Manual URL override not implemented for {self.exchange.id}.")
            elif self.exchange.id == 'bybit': # Direct manual override for Bybit if method not present
                testnet_url = self.exchange.urls.get('test', 'https://api-testnet.bybit.com')
                self.exchange.urls['api'] = testnet_url
                self.logger.info(f"Manual Bybit testnet URL override applied: {testnet_url}")
            else:
                self.logger.warning(
                    f"{NEON_YELLOW}{self.exchange.id} doesn't support set_sandbox_mode or known manual override. "
                    f"Ensure API keys are Testnet keys if using sandbox.{RESET_ALL_STYLE}"
                )

        # Cache for market info (per instance)
        self.markets_cache: Dict[str, Dict[str, Any]] = {}
        self.last_markets_update = 0 # Timestamp of last market update

        self.logger.info(f"Initialized Bybit API client (testnet={self.testnet})")

    # --- Helper methods for API interaction and error handling ---

    async def _handle_fetch_exception(
        self, e: Exception, attempt: int, total_attempts: int, item_desc: str, context_info: str
    ) -> bool:
        """
        Helper to log and determine if a fetch exception is retryable for async functions.

        Args:
            e: The exception object.
            attempt: The current retry attempt number (0-indexed).
            total_attempts: The maximum number of attempts (including the initial one).
            item_desc: A string describing the item being fetched (e.g., "price", "klines").
            context_info: A string providing context (e.g., symbol, currency).

        Returns:
            True if the exception is retryable and attempts remain, False otherwise.
        """
        is_retryable = False
        current_retry_delay = RETRY_DELAY_SECONDS # Default delay
        error_detail = str(e)
        log_level_method = self.logger.error # Default logging level

        # Common retryable errors
        if isinstance(e, (ccxt_async.NetworkError, ccxt_async.RequestTimeout, asyncio.TimeoutError)):
            log_level_method = self.logger.warning
            is_retryable = True
            msg = f"Network/Timeout error fetching {item_desc} for {context_info}"
        elif isinstance(e, (ccxt_async.RateLimitExceeded, ccxt_async.DDoSProtection)):
            log_level_method = self.logger.warning
            is_retryable = True
            msg = f"Rate limit/DDoS triggered fetching {item_desc} for {context_info}"
            # Use longer, exponential backoff for rate limits
            current_retry_delay = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS * 3, max_cap=180.0)
        # Exchange-specific errors
        elif isinstance(e, ccxt_async.ExchangeError):
            msg = f"Exchange error fetching {item_desc} for {context_info}"
            err_str_lower = error_detail.lower()
            error_code = getattr(e, 'code', None) # Get CCXT error code if available

            # Bybit specific error checks (based on common Bybit V5 issues)
            if self.exchange.id == 'bybit':
                # 10001: Illegal category - often happens during market loading or balance fetch
                if error_code == 10001 and 'category' in err_str_lower:
                     msg = f"Bybit V5 'Illegal category' error fetching {item_desc} for {context_info}"
                     log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET_ALL_STYLE}")
                     return False # Non-retryable setup issue
                # 110009: Margin account not exist - non-retryable setup issue
                elif error_code == 110009:
                     msg = f"Bybit V5 'Margin account not exist' error fetching {item_desc} for {context_info}"
                     log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET_ALL_STYLE}")
                     return False # Non-retryable setup issue
                # 110025: Position not found - expected when no position exists, not an error to log loudly
                elif error_code == 110025 and item_desc.startswith('position'):
                     self.logger.info(f"Bybit V5 'Position not found' ({error_code}) for {context_info}. This is expected if no position is open.")
                     return False # Not an error to retry
                # Other Bybit errors that might be temporary
                elif error_code in [500, 502, 503, 504]: # Standard HTTP server errors
                     log_level_method = self.logger.warning
                     is_retryable = True
                     msg = f"Bybit V5 temporary server error ({error_code}) fetching {item_desc} for {context_info}"
                else: # Other Bybit exchange errors
                     log_level_method = self.logger.error # Default to error for unknown exchange errors
                     is_retryable = False # Assume non-retryable unless specified
                     msg = f"Bybit V5 Exchange error ({error_code}) fetching {item_desc} for {context_info}"

            # Generic ExchangeError checks (apply to all exchanges)
            elif any(phrase in err_str_lower for phrase in ["symbol", "market", "not found", "invalid", "parameter", "argument"]):
                msg = f"Exchange error (invalid parameter/symbol) fetching {item_desc} for {context_info}"
                log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET_ALL_STYLE}")
                return False # Non-retryable client-side error
            else:
                # Some exchange errors might be temporary (e.g., internal server error, temporary trading ban)
                log_level_method = self.logger.warning # Downgrade to warning for potential retry
                is_retryable = True
                msg = f"Potentially temporary Exchange error fetching {item_desc} for {context_info}"

        # Authentication errors are not retryable for a single fetch attempt
        elif isinstance(e, ccxt_async.AuthenticationError):
             log_level_method = self.logger.error
             msg = f"Authentication error fetching {item_desc} for {context_info}. Check API keys/permissions."
             log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET_ALL_STYLE}")
             return False # Non-retryable

        # Handle other unexpected errors
        else:
            log_level_method = self.logger.error
            is_retryable = False
            msg = f"Unexpected error fetching {item_desc} for {context_info}"
            # Log with exc_info for unexpected errors to get traceback
            log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET_ALL_STYLE}", exc_info=True)
            return False # No retry for truly unexpected errors handled here

        # Log the error message with appropriate color and attempt info
        log_level_method(f"{NEON_YELLOW if is_retryable else NEON_RED}{msg}: {error_detail} (Attempt {attempt + 1}/{total_attempts}){RESET_ALL_STYLE}")

        # If retryable and not the last attempt, wait before retrying
        if is_retryable and attempt < total_attempts - 1:
            self.logger.warning(f"Waiting {current_retry_delay:.2f}s before retrying {item_desc} fetch for {context_info}...")
            await asyncio.sleep(current_retry_delay)

        return is_retryable # Return whether a retry should be attempted

    # --- Manual Signing and Request Methods (Optional, for endpoints not in CCXT) ---
    # Keeping these methods from the user's provided code, but focusing on CCXT async methods first.
    # These might be useful for advanced/custom API calls not covered by CCXT.

    def _sign_request(self, method: str, endpoint: str, params: Dict[str, Any] = {}) -> Dict[str, str]:
        """
        Manually signs a Bybit API request (synchronous helper).
        Note: This is for raw API calls, CCXT handles signing for its methods.
        """
        # Bybit V5 signing requires parameters to be sorted alphabetically
        # and included in the signature string along with timestamp and recvWindow.
        # This implementation is simplified and might need adjustment based on the exact endpoint.
        # CCXT's built-in signing is generally more robust.

        timestamp = str(int(time.time() * 1000))
        recv_window = "5000" # Default recvWindow, can be adjusted

        # Prepare parameters for signing
        params_to_sign = params.copy()
        params_to_sign['timestamp'] = timestamp
        params_to_sign['recvWindow'] = recv_window

        # Sort parameters alphabetically by key
        sorted_params = sorted(params_to_sign.items())

        # Create query string
        query_string = urllib.parse.urlencode(sorted_params)

        # Create signature string (timestamp + apiKey + recvWindow + query_string)
        # Note: This signature method might vary slightly by endpoint/version.
        # Bybit V5 often uses: timestamp + apiKey + recvWindow + query_string
        # Or for POST: timestamp + apiKey + recvWindow + body (JSON string)
        # This implementation assumes the query string method.
        signature_string = timestamp + self.api_key + recv_window + query_string

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(signature_string, 'utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Prepare headers
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window,
            'X-BAPI-SIGN': signature,
            'Content-Type': 'application/json' if method.upper() == 'POST' else 'application/x-www-form-urlencoded'
        }

        return headers

    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = {}) -> Any:
        """
        Makes a raw, signed request to a Bybit API endpoint (async).
        Note: Use CCXT's built-in methods whenever possible.
        """
        # This method would require an aiohttp session or similar async HTTP client.
        # Implementing a full async HTTP client here is beyond the scope, and CCXT
        # already provides this. This method is kept as a placeholder for potential
        # raw API calls if absolutely necessary, but would need a proper async
        # HTTP client implementation.

        self.logger.warning(
            f"{NEON_YELLOW}Using placeholder _make_request. Implement with a proper async HTTP client if needed for raw API calls.{RESET_ALL_STYLE}"
        )
        # Example placeholder:
        # async with aiohttp.ClientSession() as session:
        #     headers = self._sign_request(method, endpoint, params)
        #     url = f"https://api.bybit.com{endpoint}" # Adjust base URL for testnet/mainnet
        #     if method.upper() == 'GET':
        #         response = await session.get(url, headers=headers, params=params)
        #     elif method.upper() == 'POST':
        #         response = await session.post(url, headers=headers, json=params) # Assuming JSON body for POST
        #     # Handle response, check status, parse JSON, etc.
        #     return await response.json()
        raise NotImplementedError("Raw async request method (_make_request) is not fully implemented.")


    # --- Public API Interaction Methods (Using CCXT Async) ---

    async def load_markets(self) -> bool:
        """
        Loads or reloads market information from the exchange.

        Includes retry logic. Market info is cached.

        Returns:
            True if markets were loaded successfully, False otherwise.
        """
        # Use the same params as load_markets in initialize_exchange if available,
        # otherwise use a default empty dict.
        market_load_params = self.exchange.safe_value(self.exchange.options, 'loadMarketsParams', {})

        # Retry market loading as it's a critical step
        total_attempts = MAX_API_RETRIES + 1
        for attempt in range(total_attempts):
            try:
                self.logger.debug(f"Attempting to load markets (Attempt {attempt + 1}/{total_attempts}) with params: {market_load_params}")
                # Use CCXT async load_markets
                self.markets_cache = await self.exchange.load_markets(reload=True, params=market_load_params) # reload=True ensures fresh market data
                self.last_markets_update = time.time()
                self.logger.info(f"Markets loaded successfully for {self.exchange.id}.")
                return True # Success, exit retry loop
            except Exception as e:
                 # Use _handle_fetch_exception helper for consistent logging and retry logic
                 # Pass a dummy item_desc and context_info for this general fetch operation
                 if not await self._handle_fetch_exception(e, attempt, total_attempts, "markets", self.exchange.id):
                      # If _handle_fetch_exception returns False, it's a non-retryable error or max retries hit
                      self.logger.error(f"{NEON_RED}Failed to load markets after {total_attempts} attempts.{RESET_ALL_STYLE}")
                      return False
        else: # This else block executes if the loop completes without a break (i.e., all attempts failed)
             self.logger.error(f"{NEON_RED}Failed to load markets after {total_attempts} attempts.{RESET_ALL_STYLE}")
             return False

    async def get_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves and caches market information for a symbol.

        Ensures essential precision and limits fields are present with defaults.
        Adds 'is_contract' and 'amountPrecision' for easier access.
        Automatically reloads markets if info is missing or stale.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").

        Returns:
            The market information dictionary, or None on failure.
        """
        cache_key = f"{self.exchange.id}:{symbol}"
        # Check cache first
        if cache_key in self.markets_cache and (time.time() - self.last_markets_update) < 3600: # Cache for 1 hour
            self.logger.debug(f"Using cached market info for {symbol}.")
            return self.markets_cache[cache_key]

        try:
            # Ensure markets are loaded and up-to-date
            if not self.markets_cache or symbol not in self.markets_cache or (time.time() - self.last_markets_update) >= 3600:
                 self.logger.info(f"Market info for {symbol} not found in cache or cache is stale. Reloading markets...")
                 if not await self.load_markets(): # Use the instance method
                      self.logger.error(f"{NEON_RED}Failed to load markets to get info for {symbol}.{RESET_ALL_STYLE}")
                      return None

            if symbol not in self.markets_cache:
                self.logger.error(f"{NEON_RED}Market {symbol} not found on {self.exchange.id} even after attempting to load markets.{RESET_ALL_STYLE}")
                return None

            market = self.markets_cache.get(symbol)
            if not market: # Should not happen if symbol is in markets_cache, but defensive check
                self.logger.error(f"{NEON_RED}markets_cache.get({symbol}) returned None despite symbol being in cache keys.{RESET_ALL_STYLE}")
                return None

            # Ensure essential precision and limits keys exist with sane defaults if missing
            market.setdefault('precision', {})
            # Default price precision (e.g., 8 decimal places) - used if exchange doesn't provide
            market['precision'].setdefault('price', '1e-8')
            # Default amount precision (step size) - used if exchange doesn't provide
            market['precision'].setdefault('amount', '1e-8')

            market.setdefault('limits', {})
            market['limits'].setdefault('amount', {}).setdefault('min', '0') # Default min amount limit
            market['limits'].setdefault('cost', {}).setdefault('min', '0') # Default min cost limit

            # Determine if the market is a contract (future/swap) - add common checks
            market['is_contract'] = market.get('contract', False) or \
                                    market.get('type', 'unknown').lower() in ['swap', 'future', 'option', 'linear', 'inverse'] or \
                                    market.get('spot', False) is False # If not spot, assume it's a contract type


            # Calculate 'amountPrecision' (number of decimal places for amount) if not present or incorrect type
            # This is often derived from market['precision']['amount'] (step size)
            if 'amountPrecision' not in market or not isinstance(market.get('amountPrecision'), int):
                amount_step_val = market['precision'].get('amount')
                derived_precision = 8 # Default fallback if derivation fails
                if isinstance(amount_step_val, int) and amount_step_val >= 0:
                    derived_precision = amount_step_val # If it's already an integer precision
                elif isinstance(amount_step_val, (float, str, Decimal)):
                    try:
                        step = Decimal(str(amount_step_val))
                        if step > Decimal('0'):
                            # Calculate decimal places from step (e.g., 0.001 -> 3)
                            derived_precision = abs(step.normalize().as_tuple().exponent)
                    except (InvalidOperation, TypeError):
                        self.logger.warning(f"Could not derive amountPrecision from step '{amount_step_val}' for {symbol}. Using default {derived_precision}.")
                market['amountPrecision'] = derived_precision
                self.logger.debug(f"Derived amountPrecision for {symbol}: {market['amountPrecision']}")


            self.logger.debug(
                f"Market Info for {symbol}: Type={market.get('type')}, Contract={market['is_contract']}, "
                f"Price Precision Step={market['precision'].get('price')}, Amount Step={market['precision'].get('amount')}, "
                f"Amount Precision (Decimal Places)={market['amountPrecision']}"
            )
            self.markets_cache[cache_key] = market # Update cache
            return market
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error getting or processing market info for {symbol}: {e}{RESET_ALL_STYLE}", exc_info=True)
            return None

    async def check_connection(self) -> bool:
        """
        Test API connection by fetching exchange time.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            # Using fetch_time instead of fetch_status since Bybit supports it
            # fetch_time is an async method in ccxt.async_support
            time_ms = await self.exchange.fetch_time()
            if time_ms and time_ms > 0:
                 self.logger.info(f"{NEON_GREEN}API connection successful. Exchange time: {time_ms}{RESET_ALL_STYLE}")
                 return True
            else:
                 self.logger.error(f"{NEON_RED}API connection failed. fetch_time returned invalid value: {time_ms}{RESET_ALL_STYLE}")
                 return False
        except Exception as e:
            self.logger.error(f"{NEON_RED}API connection failed: {e}{RESET_ALL_STYLE}", exc_info=True)
            return False

    async def fetch_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Fetches the current market price for a symbol using fetch_ticker.

        Prioritizes mid-price (from bid/ask), then last, close, ask, bid.
        Includes retry logic.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").

        Returns:
            The current price as a Decimal, or None if fetching fails after retries.
        """
        attempts = 0
        total_attempts = MAX_API_RETRIES + 1
        while attempts < total_attempts:
            try:
                self.logger.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1}/{total_attempts})")
                ticker = await self.exchange.fetch_ticker(symbol)

                if not ticker:
                     self.logger.warning(f"fetch_ticker returned None or empty for {symbol} on attempt {attempts + 1}.")
                     raise ccxt_async.ExchangeError("Empty ticker data.") # Trigger retry

                # Prioritize different price fields from the ticker
                price_sources = []
                # Ensure bid/ask are valid numbers before calculating mid-price
                bid_val = ticker.get('bid')
                ask_val = ticker.get('ask')

                if bid_val is not None and ask_val is not None:
                    try:
                        bid = Decimal(str(bid_val))
                        ask = Decimal(str(ask_val))
                        if bid > Decimal('0') and ask > Decimal('0') and ask >= bid:
                            price_sources.append((bid + ask) / Decimal('2')) # Mid-price
                    except (InvalidOperation, TypeError):
                        self.logger.debug(f"Could not parse bid/ask for mid-price for {symbol}. Trying other sources.", exc_info=True)

                # Order of preference for fallback: last, close, ask, bid (after mid-price attempt)
                for key in ['last', 'close', 'ask', 'bid']:
                    # Only add if not None and not already processed as part of bid/ask for mid-price
                    # Check if the key exists in the ticker and its value is not None
                    if ticker.get(key) is not None and (key != 'bid' and key != 'ask' or bid_val is None or ask_val is None):
                         # Add the raw value first, conversion to Decimal happens next
                         price_sources.append(ticker[key])

                for price_val_raw in price_sources:
                    if price_val_raw is not None:
                        try:
                            price_dec = Decimal(str(price_val_raw))
                            if price_dec > Decimal('0'):
                                self.logger.debug(f"Price for {symbol} obtained: {price_dec}")
                                return price_dec
                        except (InvalidOperation, TypeError):
                            # Conversion failed, try the next source
                            self.logger.debug(f"Could not convert raw price value '{price_val_raw}' from key to Decimal for {symbol}. Trying next source.")
                            continue # Try next price source

                self.logger.warning(f"No valid price (mid, last, close, bid, ask) found in ticker for {symbol} on attempt {attempts + 1}. Ticker keys: {list(ticker.keys()) if isinstance(ticker, dict) else 'N/A'}")
                # If ticker was fetched but no usable price found, raise to trigger retry
                raise ccxt_async.ExchangeError("No valid price found in ticker data.")

            except Exception as e:
                # Use _handle_fetch_exception helper for consistent logging and retry logic
                if not await self._handle_fetch_exception(e, attempts, total_attempts, f"price for {symbol}", symbol):
                    return None # Non-retryable error or max retries exceeded

            attempts += 1 # Increment attempt only if _handle_fetch_exception indicated retry

        self.logger.error(f"Failed to fetch price for {symbol} after {total_attempts} attempts.")
        return None

    async def fetch_klines(
        self, symbol: str, timeframe: str, limit: int = 250
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data (candlesticks) for a symbol and timeframe.

        Includes retry logic and data validation/cleaning.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            timeframe: The timeframe for the klines (e.g., "5m", "1h").
            limit: The maximum number of klines to fetch.

        Returns:
            A pandas DataFrame with cleaned and validated kline data, or an empty DataFrame on failure.
        """
        if not self.exchange.has['fetchOHLCV']:
            self.logger.error(f"Exchange {self.exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        total_attempts = MAX_API_RETRIES + 1
        for attempt in range(total_attempts):
            try:
                self.logger.debug(f"Fetching klines for {symbol} (Timeframe: {timeframe}, Limit: {limit}) (Attempt {attempt + 1}/{total_attempts})")
                # Add Bybit V5 specific params if needed for fetchOHLCV
                ohlcv_params = {}
                if self.exchange.id == 'bybit':
                    # Bybit V5 fetchOHLCV needs category
                    market_info = await self.get_market_info(symbol) # Use instance method
                    if market_info:
                        market_type = market_info.get('type', '').lower()
                        if market_type in ['linear', 'inverse', 'spot']:
                             ohlcv_params['category'] = market_type
                        elif market_info.get('linear', False):
                             ohlcv_params['category'] = 'linear'
                        elif market_info.get('inverse', False):
                             ohlcv_params['category'] = 'inverse'
                        elif self.exchange.options.get('defaultType') == 'unified':
                             # For unified, fetchOHLCV usually needs linear/inverse/spot category
                             # Default to linear if not specified elsewhere
                             ohlcv_params['category'] = 'linear'
                             self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit V5 category for fetchOHLCV from market info type '{market_type}' or defaultType 'unified'. Defaulting to 'linear'.{RESET_ALL_STYLE}")
                        else:
                             self.logger.warning(f"{NEON_YELLOW}Could not determine Bybit V5 category for fetchOHLCV for {symbol}. Proceeding without category param.{RESET_ALL_STYLE}")

                    # Add recvWindow if not present in params or options
                    if 'recvWindow' not in ohlcv_params and 'recvWindow' not in self.exchange.options.get('options', {}):
                        ohlcv_params['recvWindow'] = 5000 # milliseconds
                        self.logger.debug("Setting recvWindow param for Bybit fetchOHLCV.")


                ohlcv_data = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=ohlcv_params)

                if ohlcv_data and isinstance(ohlcv_data, list) and len(ohlcv_data) > 0 and \
                   all(isinstance(row, list) and len(row) >= 6 for row in ohlcv_data): # Check basic structure
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                    # Convert timestamp to datetime, normalize to UTC, then remove tz for consistency
                    # Use errors='coerce' to turn unparseable dates into NaT
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
                    df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None) # Remove timezone info after converting to UTC
                    df.set_index('timestamp', inplace=True)

                    # Convert OHLCV columns to Decimal, handling potential errors
                    # Use .astype(object) first to prevent issues with mixed types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        try:
                            df[col] = df[col].astype(object).apply(
                                lambda x: Decimal(str(x)) if pd.notna(x) and str(x).strip() != "" else None
                            )
                        except Exception as conv_err: # Catch broader exceptions during apply
                            self.logger.warning(
                                f"Could not convert column '{col}' to Decimal for {symbol} due to: {conv_err}. "
                                f"Falling back to pd.to_numeric for this column, data might lose precision or be invalid."
                            )
                            # Fallback to numeric conversion, coercing errors to NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Drop rows with NaN in critical OHLC columns after conversion attempts
                    # Use .copy() to avoid SettingWithCopyWarning
                    df_cleaned = df.dropna(subset=['open', 'high', 'low', 'close'], how='any').copy()

                    # Filter out candles with non-positive close price or negative volume
                    # Ensure we are checking Decimal values if conversion was successful, otherwise use numeric check
                    df_cleaned = df_cleaned[
                        df_cleaned.apply(
                            lambda row: (isinstance(row['close'], Decimal) and row['close'] > Decimal('0')) or
                                        (pd.notna(row['close']) and row['close'] > 0), axis=1
                        )
                    ].copy()
                     # Allow volume == 0, but not negative
                    df_cleaned = df_cleaned[
                        df_cleaned.apply(
                            lambda row: (isinstance(row['volume'], Decimal) and row['volume'] >= Decimal('0')) or
                                        (pd.notna(row['volume']) and row['volume'] >= 0), axis=1
                        )
                    ].copy()


                    if df_cleaned.empty:
                        self.logger.warning(f"Klines data for {symbol} {timeframe} is empty after cleaning and validation.")
                        # Consider this a fetch failure to allow retry if appropriate
                        raise ccxt_async.ExchangeError("Cleaned kline data is empty.")

                    df_cleaned.sort_index(inplace=True) # Ensure chronological order
                    self.logger.info(f"Successfully fetched and processed {len(df_cleaned)} klines for {symbol} {timeframe}.")
                    return df_cleaned
                else:
                    self.logger.warning(
                        f"Received empty or invalid kline data structure for {symbol} {timeframe}. "
                        f"Data: {ohlcv_data[:2] if ohlcv_data else 'None'}... (Attempt {attempt + 1})"
                    )
                    raise ccxt_async.ExchangeError("Empty or invalid kline data structure from exchange.")

            except Exception as e:
                # Use _handle_fetch_exception helper
                if not await self._handle_fetch_exception(e, attempt, total_attempts, f"klines for {symbol} {timeframe}", symbol):
                    return pd.DataFrame() # Non-retryable or max retries hit

        self.logger.error(f"Failed to fetch klines for {symbol} {timeframe} after {total_attempts} attempts.")
        return pd.DataFrame()


    async def fetch_orderbook(
        self, symbol: str, limit: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches the order book for a symbol.

        Includes retry logic and basic structure validation.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            limit: The number of bids and asks to fetch.

        Returns:
            The order book dictionary, or None on failure.
        """
        if not self.exchange.has['fetchOrderBook']:
            self.logger.error(f"Exchange {self.exchange.id} does not support fetchOrderBook.")
            return None

        attempts = 0
        total_attempts = MAX_API_RETRIES + 1
        while attempts < total_attempts:
            try:
                self.logger.debug(f"Fetching order book for {symbol} (Limit: {limit}) (Attempt {attempts + 1}/{total_attempts})")

                # Add Bybit V5 specific params if needed for fetchOrderBook
                orderbook_params = {}
                if self.exchange.id == 'bybit':
                    # Bybit V5 fetchOrderBook needs category
                    market_info = await self.get_market_info(symbol) # Use instance method
                    if market_info:
                        market_type = market_info.get('type', '').lower()
                        if market_type in ['linear', 'inverse', 'spot']:
                             orderbook_params['category'] = market_type
                        elif market_info.get('linear', False):
                             orderbook_params['category'] = 'linear'
                        elif market_info.get('inverse', False):
                             orderbook_params['category'] = 'inverse'
                        elif self.exchange.options.get('defaultType') == 'unified':
                             # For unified, fetchOrderBook usually needs linear/inverse/spot category
                             # Default to linear if not specified elsewhere
                             orderbook_params['category'] = 'linear'
                             self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit V5 category for fetchOrderBook from market info type '{market_type}' or defaultType 'unified'. Defaulting to 'linear'.{RESET_ALL_STYLE}")
                        else:
                             self.logger.warning(f"{NEON_YELLOW}Could not determine Bybit V5 category for fetchOrderBook for {symbol}. Proceeding without category param.{RESET_ALL_STYLE}")

                    # Add recvWindow if not present in params or options
                    if 'recvWindow' not in orderbook_params and 'recvWindow' not in self.exchange.options.get('options', {}):
                        orderbook_params['recvWindow'] = 5000 # milliseconds
                        self.logger.debug("Setting recvWindow param for Bybit fetchOrderBook.")


                order_book = await self.exchange.fetch_order_book(symbol, limit=limit, params=orderbook_params)

                if order_book and isinstance(order_book, dict) and \
                   'bids' in order_book and isinstance(order_book['bids'], list) and \
                   'asks' in order_book and isinstance(order_book['asks'], list):
                    if not order_book['bids'] and not order_book['asks']:
                        self.logger.warning(f"Order book for {symbol} fetched but bids and asks arrays are empty.")
                    # Basic structure is valid, return it
                    return order_book
                else:
                    self.logger.warning(
                        f"Invalid order book structure received for {symbol} on attempt {attempts + 1}. "
                        f"Data: {str(order_book)[:200]}... (Attempt {attempts + 1})" # Log snippet of problematic data
                    )
                    raise ccxt_async.ExchangeError("Invalid order book structure received.")
            except Exception as e:
                # Use _handle_fetch_exception helper
                if not await self._handle_fetch_exception(e, attempt, total_attempts, f"orderbook for {symbol}", symbol):
                    return None
            attempts += 1 # Increment attempt only if _handle_fetch_exception indicated retry

        self.logger.error(f"Failed to fetch order book for {symbol} after {total_attempts} attempts.")
        return None

    async def fetch_balance(
        self, currency: str, params: Optional[Dict] = None
    ) -> Optional[Decimal]:
        """
        Fetches the available balance for a specific currency.

        Includes retry logic and specific handling for Bybit V5 account types.

        Args:
            currency: The currency code (e.g., "USDT").
            params: Optional additional parameters for the fetch_balance call.

        Returns:
            The available balance as a Decimal, or None on failure.
        """
        request_params = params.copy() if params is not None else {} # Use a copy to avoid modifying caller's dict

        # Specific handling for Bybit account types if not provided in params, using exchange options default
        if self.exchange.id == 'bybit':
            # Bybit V5 needs accountType parameter for fetchBalance
            # Prioritize accountType from request_params, then infer from defaultType
            if 'accountType' not in request_params:
                default_type = self.exchange.options.get('defaultType', '').upper()
                if default_type == 'UNIFIED':
                    request_params['accountType'] = 'UNIFIED'
                elif default_type in ['LINEAR', 'INVERSE', 'CONTRACT']:
                     request_params['accountType'] = 'CONTRACT' # CONTRACT covers linear/inverse for balance
                elif default_type == 'SPOT':
                     request_params['accountType'] = 'SPOT'
                else:
                     # Fallback if defaultType is ambiguous, try UNIFIED as it's common for V5
                     request_params['accountType'] = 'UNIFIED'
                     self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit accountType for fetchBalance from defaultType '{self.exchange.options.get('defaultType')}'. Defaulting to 'UNIFIED'.{RESET_ALL_STYLE}")

        # Add recvWindow if not present in params or options
        if 'recvWindow' not in request_params and 'recvWindow' not in self.exchange.options.get('options', {}):
            request_params['recvWindow'] = 5000 # milliseconds
            self.logger.debug("Setting recvWindow param for Bybit balance fetch.")


        attempts = 0
        total_attempts = MAX_API_RETRIES + 1
        while attempts < total_attempts:
            try:
                log_params = request_params.copy()
                # Mask potentially sensitive info in params if any (unlikely for balance fetch params)
                # if 'api_key' in log_params: log_params['api_key'] = '***'

                self.logger.debug(
                    f"Fetching balance for {currency} (Attempt {attempts + 1}/{total_attempts}). "
                    f"Params: {log_params if log_params else 'None'}"
                )
                balance_info = await self.exchange.fetch_balance(params=request_params)

                if balance_info:
                    # Try to get currency-specific balance data first from the main balance dictionary
                    currency_data = balance_info.get(currency.upper()) # Ensure currency code is uppercase
                    available_balance_str = None

                    if currency_data and currency_data.get('free') is not None:
                        available_balance_str = str(currency_data['free'])
                    elif currency_data and currency_data.get('total') is not None:
                        # Use 'total' if 'free' is not available, but log a warning
                        available_balance_str = str(currency_data['total'])
                        self.logger.warning(
                            f"Using 'total' balance for {currency} as 'free' is unavailable. "
                            f"This might include locked funds."
                        )
                    # Fallback for structures where 'free' is a top-level dict (less common in V5 but possible)
                    elif 'free' in balance_info and isinstance(balance_info['free'], dict) and \
                         balance_info['free'].get(currency.upper()) is not None:
                        available_balance_str = str(balance_info['free'][currency.upper()])

                    if available_balance_str is not None:
                        try:
                            final_balance = Decimal(available_balance_str)
                            if final_balance >= Decimal('0'): # Balance should be non-negative
                                self.logger.info(f"Available {currency} balance: {final_balance:.8f}")
                                return final_balance
                            else:
                                self.logger.error(f"Parsed balance for {currency} is negative ({final_balance}). This is unusual.")
                                # Treat negative balance as an error state for the fetch
                                raise ccxt_async.ExchangeError(f"Negative balance received for {currency}: {final_balance}")
                        except InvalidOperation:
                            self.logger.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}.")
                            # Treat conversion failure as an error for the fetch
                            raise ccxt_async.ExchangeError(f"Failed to parse balance string: {available_balance_str}")
                    else:
                        self.logger.error(
                            f"Could not determine free balance for {currency}. "
                            f"Relevant balance keys: {list(balance_info.keys() if isinstance(balance_info, dict) else [])}. "
                            f"Currency data: {currency_data}. Full balance info keys: {list(balance_info.keys())}"
                        )
                        # If balance info is present but currency data isn't found or is incomplete
                        raise ccxt_async.ExchangeError(f"Balance data for {currency} not found or incomplete in response.")
                else:
                    self.logger.error(f"Balance info response was None or empty on attempt {attempts + 1}.")
                    # If balance info is None or empty
                    raise ccxt_async.ExchangeError("Empty balance info response.")

            except Exception as e:
                # Use _handle_fetch_exception helper
                if not await self._handle_fetch_exception(e, attempts, total_attempts, f"balance for {currency}", currency):
                    return None
            attempts += 1 # Increment attempt only if _handle_fetch_exception indicated retry

        self.logger.error(f"Failed to fetch balance for {currency} after {total_attempts} attempts.")
        return None


    async def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches and standardizes information about an open position for a symbol.

        Handles variations in exchange responses and filters for active positions.
        Includes specific logic for Bybit V5.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").

        Returns:
            A standardized dictionary for the open position, or None if no active position is found.
        """
        # Check if the exchange supports fetching positions
        if not self.exchange.has.get('fetchPositions'):
            self.logger.warning(f"Exchange {self.exchange.id} does not support fetchPositions.")
            return None

        market_info = await self.get_market_info(symbol) # Use instance method
        if not market_info:
             self.logger.error(f"{NEON_RED}Cannot get position info for {symbol}: Market info not available.{RESET_ALL_STYLE}")
             return None

        # Prepare parameters for fetching positions
        # Prioritize fetchPositionsParams from exchange options, then add Bybit V5 defaults
        fetch_pos_params = self.exchange.safe_value(self.exchange.options, 'fetchPositionsParams', {}).copy()

        # Bybit V5 specific: accountType and symbol are often required
        if self.exchange.id == 'bybit':
            if 'accountType' not in fetch_pos_params:
                default_type = self.exchange.options.get('defaultType', '').upper()
                if default_type == 'UNIFIED':
                    fetch_pos_params['accountType'] = 'UNIFIED'
                elif default_type in ['LINEAR', 'INVERSE', 'CONTRACT']:
                     fetch_pos_params['accountType'] = 'CONTRACT'
                elif default_type == 'SPOT':
                     fetch_pos_params['accountType'] = 'SPOT'
                else:
                     # Fallback for ambiguous defaultType
                     fetch_pos_params['accountType'] = 'UNIFIED'
                     self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit accountType for fetchPositions from defaultType '{self.exchange.options.get('defaultType')}'. Defaulting to 'UNIFIED'.{RESET_ALL_STYLE}")

            # Bybit V5 fetchPositions often requires the exchange-specific symbol ID
            if 'symbol' not in fetch_pos_params:
                 fetch_pos_params['symbol'] = market_info.get('id')
                 if not fetch_pos_params['symbol']:
                      self.logger.warning(f"{NEON_YELLOW}Could not get Bybit V5 exchange-specific symbol ID for fetchPositions.{RESET_ALL_STYLE}")
                      # Remove symbol from params if not found, let API decide
                      fetch_pos_params.pop('symbol', None)

            # Add recvWindow if not present in params or options
            if 'recvWindow' not in fetch_pos_params and 'recvWindow' not in self.exchange.options.get('options', {}):
                fetch_pos_params['recvWindow'] = 5000 # milliseconds
                self.logger.debug("Setting recvWindow param for Bybit fetchPositions.")


        positions: List[Dict[str, Any]] = []
        total_attempts = MAX_API_RETRIES + 1
        for attempt in range(total_attempts):
            try:
                self.logger.debug(f"Fetching position for {symbol} (Attempt {attempt + 1}/{total_attempts}) with params {fetch_pos_params}...")
                # Try fetching positions for the specific symbol first if CCXT supports it with params
                # Note: Some exchanges might return all positions even when a symbol is requested in params.
                # CCXT's fetch_positions([symbol], params) is the ideal way if supported.
                # If it fails with ArgumentsRequired, we fall back to fetching all.

                # Check if exchange explicitly supports fetching positions by symbol list
                if self.exchange.has.get('fetchPositions', 'emulated') is True: # True means native support
                     fetched_positions_raw = await self.exchange.fetch_positions([symbol], params=fetch_pos_params)
                     # Filter again just in case the exchange returned all positions
                     positions = [p for p in fetched_positions_raw if p.get('symbol') == symbol]
                else: # Emulated or unknown support, fetch all and filter
                     self.logger.debug(f"Exchange {self.exchange.id} fetchPositions may not natively support symbol list. Fetching all positions and filtering.")
                     all_positions = await self.exchange.fetch_positions(params=fetch_pos_params)
                     # Filter by symbol or market_id from the 'info' field for robustness
                     market_id = market_info.get('id') # Exchange-specific market ID
                     positions = [
                         p for p in all_positions if p.get('symbol') == symbol or
                         (market_id and p.get('info') and p['info'].get('symbol') == market_id)
                     ]

                # If we got here without exception, the fetch was successful (even if positions list is empty)
                break # Exit retry loop

            except ccxt_async.ArgumentsRequired: # If fetchPositions([symbols]) is not supported, this might be raised
                self.logger.debug(
                    f"fetchPositions for {self.exchange.id} with symbol argument failed or is not supported. "
                    f"Attempting to fetch all positions and filter for {symbol}."
                )
                try:
                    # Pass params to fetch_positions (all)
                    all_positions = await self.exchange.fetch_positions(params=fetch_pos_params)
                    # Filter by symbol or market_id from the 'info' field for robustness
                    market_id = market_info.get('id') # Exchange-specific market ID
                    positions = [
                        p for p in all_positions if p.get('symbol') == symbol or
                        (market_id and p.get('info') and p['info'].get('symbol') == market_id)
                    ]
                    break # Success fetching all, exit retry loop
                except Exception as e_all:
                    # Handle exceptions during the fallback 'fetch all' attempt
                    if not await self._handle_fetch_exception(e_all, attempt, total_attempts, f"all positions (fallback)", self.exchange.id):
                         return None # Non-retryable or max retries hit for fallback
            except Exception as e:
                # Use _handle_fetch_exception helper for other exceptions
                if not await self._handle_fetch_exception(e, attempt, total_attempts, f"position for {symbol}", symbol):
                    return None # Non-retryable or max retries hit

        else: # This else block executes if the retry loop completes without a break (all attempts failed)
            self.logger.error(f"Failed to fetch position for {symbol} after {total_attempts} attempts.")
            return None # Return None if fetching completely failed

        # --- Process fetched positions to find the active one ---
        if not positions:
            self.logger.info(f"No position data structures returned or matched for {symbol}.")
            return None

        active_position_data = None
        # Determine a small threshold based on amount precision to filter out dust positions
        # Use market_info['amountPrecision'] (decimal places) to derive a robust threshold
        amount_precision_places = market_info.get('amountPrecision', 8) # Default to 8 if not found
        try:
            # Threshold is 1/100th of the smallest possible step size (10^-amountPrecision)
            size_threshold = Decimal(f'1e-{amount_precision_places}') / Decimal('100')
            if size_threshold <= Decimal('0'): size_threshold = Decimal('1e-10') # Ensure positive threshold, very small
        except InvalidOperation:
            size_threshold = Decimal('1e-10') # Fallback if amount_precision is invalid


        for pos_data in positions:
            # Try to get position size from common fields ('contracts', 'info.size', 'info.qty', 'size')
            # Use safe_value to handle nested 'info' dict and potential missing keys
            size_str = self.exchange.safe_value(pos_data, 'contracts') or \
                       self.exchange.safe_value(pos_data, ['info', 'size']) or \
                       self.exchange.safe_value(pos_data, ['info', 'qty']) or \
                       self.exchange.safe_value(pos_data, 'size') # Some exchanges might use 'size' directly

            if size_str is None:
                self.logger.debug(f"Position data for {symbol} missing size information. Skipping: {pos_data}")
                continue # Skip position data if size is missing

            try:
                pos_size_dec = Decimal(str(size_str))

                # Bybit V5 specific: 'positionSide' can be 'None' for closed/zero positions in hedge mode
                bybit_v5_pos_side_info = self.exchange.safe_value(pos_data, ['info', 'positionSide'], '').lower()
                if self.exchange.id == 'bybit' and bybit_v5_pos_side_info == 'none' and abs(pos_size_dec) <= size_threshold:
                    self.logger.debug(f"Skipping Bybit V5 position with positionSide 'None' and size <= threshold for {symbol}.")
                    continue # Skip Bybit "None" side positions that are effectively zero

                # Check if position size is greater than the threshold (active position)
                if abs(pos_size_dec) > size_threshold:
                    active_position_data = pos_data.copy() # Work with a copy to avoid modifying original list item

                    # Standardize 'contractsDecimal' to be positive absolute size
                    active_position_data['contractsDecimal'] = abs(pos_size_dec)

                    # Standardize 'side' ('long' or 'short')
                    current_side = active_position_data.get('side', '').lower()
                    if not current_side or current_side == 'none': # Infer if 'side' is missing or 'None'
                        # Infer from Bybit V5 'positionSide' if available
                        if self.exchange.id == 'bybit' and bybit_v5_pos_side_info in ['buy', 'sell']:
                            current_side = 'long' if bybit_v5_pos_side_info == 'buy' else 'short'
                        # Infer from position size sign if side is still not determined
                        elif pos_size_dec > size_threshold: # Positive size implies long (most exchanges)
                            current_side = 'long'
                        elif pos_size_dec < -size_threshold: # Negative size implies short (some exchanges, e.g., Bybit older API)
                            current_side = 'short'
                        else:
                            # Should not happen if abs(pos_size_dec) > size_threshold, but defensive
                            self.logger.warning(f"Could not determine side for active position with size {pos_size_dec} for {symbol}. Skipping.")
                            continue # Ambiguous side, skip this position data
                    active_position_data['side'] = current_side

                    # Entry Price - map common fields to 'entryPriceDecimal'
                    ep_str = self.exchange.safe_value(active_position_data, 'entryPrice') or \
                             self.exchange.safe_value(active_position_data, ['info', 'avgPrice'])
                    active_position_data['entryPriceDecimal'] = Decimal(str(ep_str)) if ep_str is not None and str(ep_str).strip() else None

                    # Map various potential field names to standardized Decimal keys
                    field_map = {
                        'markPriceDecimal': ['markPrice'],
                        'liquidationPriceDecimal': ['liquidationPrice', 'liqPrice'],
                        'unrealizedPnlDecimal': ['unrealizedPnl', 'unrealisedPnl', 'pnl', ('info', 'unrealisedPnl')],
                        'stopLossPriceDecimal': ['stopLoss', 'stopLossPrice', 'slPrice', ('info', 'stopLoss')],
                        'takeProfitPriceDecimal': ['takeProfit', 'takeProfitPrice', 'tpPrice', ('info', 'takeProfit')],
                        # Bybit V5 TSL fields: trailingStop is the distance, activePrice is the trigger price
                        'trailingStopDistanceDecimal': [('info', 'trailingStop'), ('info', 'trailing_stop')], # Distance value
                        'trailingStopActivationPriceDecimal': [('info', 'activePrice'), ('info', 'triggerPrice'), ('info', 'trailing_trigger_price')] # Activation price
                    }
                    for dec_key, str_keys_list in field_map.items():
                        val_str = None
                        for sk_item in str_keys_list:
                            if isinstance(sk_item, tuple): # e.g., ('info', 'someKey')
                                val_str = self.exchange.safe_value(active_position_data, sk_item)
                            else:
                                val_str = self.exchange.safe_value(active_position_data, sk_item)
                            if val_str is not None: break # Found a value

                        if val_str is not None and str(val_str).strip(): # Ensure not empty string
                            # For SL/TP, "0" often means not set; treat as None for consistency
                            # Also handle potential 'null' string or actual None from API
                            if str(val_str).strip() in ['0', 'null'] and dec_key in ['stopLossPriceDecimal', 'takeProfitPriceDecimal', 'trailingStopDistanceDecimal', 'trailingStopActivationPriceDecimal']:
                                active_position_data[dec_key] = None
                            else:
                                try:
                                    active_position_data[dec_key] = Decimal(str(val_str))
                                except (InvalidOperation, TypeError):
                                    # Log conversion failure but continue
                                    self.logger.debug(f"Could not convert value '{val_str}' for key '{dec_key}' to Decimal for {symbol}.", exc_info=True)
                                    active_position_data[dec_key] = None # Failed conversion
                        else:
                            active_position_data[dec_key] = None # Not found, empty string, or None

                    # Timestamp (ms) - map common fields to 'timestamp_ms'
                    ts_raw = self.exchange.safe_value(active_position_data, 'timestamp') or \
                             self.exchange.safe_value(active_position_data, ['info', 'updatedTime']) or \
                             self.exchange.safe_value(active_position_data, ['info', 'updated_at']) or \
                             self.exchange.safe_value(active_position_data, ['info', 'createTime'])

                    # Convert to integer milliseconds if possible
                    active_position_data['timestamp_ms'] = None
                    if ts_raw is not None:
                        try:
                            # Handle potential string timestamps, float timestamps, or integer timestamps
                            active_position_data['timestamp_ms'] = int(float(str(ts_raw)))
                        except (ValueError, TypeError):
                             self.logger.debug(f"Could not convert timestamp '{ts_raw}' to integer milliseconds for {symbol}.", exc_info=True)
                             active_position_data['timestamp_ms'] = None


                    # Leverage - map common fields to 'leverageDecimal'
                    leverage_raw = self.exchange.safe_value(active_position_data, 'leverage') or \
                                   self.exchange.safe_value(active_position_data, ['info', 'leverage'])

                    active_position_data['leverageDecimal'] = None
                    if leverage_raw is not None:
                        try:
                            leverage_dec = Decimal(str(leverage_raw))
                            if leverage_dec > Decimal('0'):
                                active_position_data['leverageDecimal'] = leverage_dec
                        except (InvalidOperation, TypeError):
                            self.logger.debug(f"Could not convert leverage '{leverage_raw}' to Decimal for {symbol}.", exc_info=True)
                            active_position_data['leverageDecimal'] = None


                    # Break the loop once an active position is found and processed
                    break
            except (InvalidOperation, ValueError, TypeError) as e:
                self.logger.warning(f"Error parsing position data for {symbol}: {e}. Data: {pos_data}", exc_info=True)
                continue # Try next position data if parsing fails

        if active_position_data:
            self.logger.info(
                f"Active {active_position_data.get('side','N/A').upper()} position found for {symbol}: "
                f"Size={active_position_data.get('contractsDecimal','N/A')}, "
                f"Entry={active_position_data.get('entryPriceDecimal','N/A')}, "
                f"Leverage={active_position_data.get('leverageDecimal','N/A')}"
            )
            return active_position_data

        self.logger.info(f"No active open position found for {symbol} after filtering (size > {size_threshold:.8f}).")
        return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Sets the leverage for a contract symbol.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            leverage: The desired leverage as an integer.

        Returns:
            True if the leverage setting was successful or already set, False otherwise.
        """
        market_info = await self.get_market_info(symbol) # Use instance method
        if not market_info:
             self.logger.error(f"{NEON_RED}Cannot set leverage for {symbol}: Market info not available.{RESET_ALL_STYLE}")
             return False

        if not market_info.get('is_contract', False):
            self.logger.info(f"Leverage setting skipped for {symbol} as it's not a contract market.")
            return True # No action needed, considered success

        if not (isinstance(leverage, int) and leverage > 0):
            self.logger.warning(f"{NEON_YELLOW}Invalid leverage value {leverage} for {symbol}. Must be a positive integer.{RESET_ALL_STYLE}")
            return False

        if not (hasattr(self.exchange, 'set_leverage') and callable(self.exchange.set_leverage)):
            self.logger.error(f"{NEON_RED}Exchange {self.exchange.id} does not support set_leverage method via CCXT.{RESET_ALL_STYLE}")
            return False

        self.logger.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        # Get leverage params from config if available, merge with default Bybit V5 params
        # Prioritize setLeverageParams from exchange options
        leverage_params = self.exchange.safe_value(self.exchange.options, 'setLeverageParams', {}).copy()

        # Bybit V5 might require buyLeverage and sellLeverage for unified margin, or for hedge mode positions.
        # For one-way mode on derivatives, setting leverage for the symbol is usually enough via CCXT's standard call.
        # CCXT's set_leverage should handle underlying requirements, but explicitly setting for Unified
        if self.exchange.id == 'bybit':
            # For Bybit V5, set_leverage typically uses the v5PrivatePostPositionSetLeverage endpoint
            # This endpoint requires category, symbol, buyLeverage, sellLeverage
            # Ensure category is set
            if 'category' not in leverage_params:
                market_type = market_info.get('type', '').lower()
                if market_type in ['linear', 'inverse', 'spot']:
                    leverage_params['category'] = market_type
                elif market_info.get('linear', False):
                    leverage_params['category'] = 'linear'
                elif market_info.get('inverse', False):
                    leverage_params['category'] = 'inverse'
                elif self.exchange.options.get('defaultType') == 'unified':
                    # For unified, set leverage needs linear/inverse category
                    leverage_params['category'] = 'linear' # Default to linear
                    self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit V5 category for setLeverage from market info type '{market_type}' or defaultType 'unified'. Defaulting to 'linear'.{RESET_ALL_STYLE}")
                else:
                     self.logger.warning(f"{NEON_YELLOW}Could not determine Bybit V5 category for setLeverage for {symbol}. Proceeding without category param.{RESET_ALL_STYLE}")

            # Ensure symbol ID is set (Bybit V5 requires exchange-specific ID)
            if 'symbol' not in leverage_params:
                 leverage_params['symbol'] = market_info.get('id')
                 if not leverage_params['symbol']:
                      self.logger.warning(f"{NEON_YELLOW}Could not get Bybit V5 exchange-specific symbol ID for setLeverage.{RESET_ALL_STYLE}")
                      # Remove symbol from params if not found, let API decide
                      leverage_params.pop('symbol', None)

            # Bybit V5 requires buyLeverage and sellLeverage as strings
            leverage_params['buyLeverage'] = str(leverage)
            leverage_params['sellLeverage'] = str(leverage)

            # Add recvWindow if not present in params or options
            if 'recvWindow' not in leverage_params and 'recvWindow' not in self.exchange.options.get('options', {}):
                leverage_params['recvWindow'] = 5000 # milliseconds
                self.logger.debug("Setting recvWindow param for Bybit setLeverage.")

        log_params = leverage_params.copy()
        # Mask sensitive fields if they exist (unlikely for leverage params)

        self.logger.debug(f"Set leverage API call for {symbol}: params={log_params}")

        try:
            # CCXT's set_leverage method is designed to abstract the underlying API calls.
            # We pass the leverage and symbol, and the pre-prepared params.
            response = await self.exchange.set_leverage(leverage, symbol, params=leverage_params)
            self.logger.debug(f"Set leverage raw response for {symbol}: {response}")

            # Specific handling for Bybit V5 response (check retCode)
            if self.exchange.id == 'bybit' and isinstance(response, dict):
                ret_code = response.get('retCode')
                ret_msg = response.get('retMsg', '').lower()
                if ret_code == 0:
                    self.logger.info(f"{NEON_GREEN}Leverage for {symbol} successfully set to {leverage}x (Bybit).{RESET_ALL_STYLE}")
                    return True
                # Bybit: 110043 means "Leverage not modified"
                elif ret_code == 110043 or "leverage not modified" in ret_msg or "same leverage" in ret_msg:
                    self.logger.info(f"Leverage for {symbol} was already {leverage}x (Bybit: {ret_code} - {ret_msg}).")
                    return True
                else:
                    self.logger.error(f"{NEON_RED}Bybit error setting leverage for {symbol}: {ret_msg} (Code: {ret_code}){RESET_ALL_STYLE}")
                    return False

            # Generic success: if no exception and not a specific Bybit failure check above
            self.logger.info(f"{NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Generic CCXT response).{RESET_ALL_STYLE}")
            return True
        except ccxt_async.ExchangeError as e:
            err_str, code = str(e).lower(), getattr(e, 'code', None)
            # Check if error message indicates leverage was already set (common for some exchanges)
            if "leverage not modified" in err_str or "no change" in err_str or \
               (self.exchange.id == 'bybit' and code == 110043): # Bybit: 110043
                # FIX: Add the indented block here
                self.logger.info(f"Leverage for {symbol} already {leverage}x (Confirmed by error: {e}).")
                return True
            self.logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol} to {leverage}x: {e} (Code: {code}){RESET_ALL_STYLE}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error setting leverage for {symbol} to {leverage}x: {e}{RESET_ALL_STYLE}", exc_info=True)
        return False

    async def place_trade(
        self, symbol: str, trade_signal: str, # "BUY" or "SELL"
        position_size: Decimal, order_type: str = 'market',
        limit_price: Optional[Decimal] = None, reduce_only: bool = False,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]: # Returns the CCXT order object if successful
        """
        Places a trade order (market or limit) to open or close a position.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            trade_signal: The intended action ("BUY" or "SELL").
            position_size: The size of the position to open/close as a Decimal.
            order_type: The type of order ('market' or 'limit').
            limit_price: The price for a limit order (required if order_type is 'limit').
            reduce_only: If True, the order will only reduce an existing position.
            params: Optional additional parameters for the create_order call.

        Returns:
            The CCXT order dictionary if the order was placed successfully, None otherwise.
        """
        side = 'buy' if trade_signal.upper() == "BUY" else 'sell'
        action_description = "Reduce-Only" if reduce_only else "Open/Increase"

        market_info = await self.get_market_info(symbol) # Use instance method
        if not market_info:
             self.logger.error(f"{NEON_RED}Trade aborted for {symbol}: Market info not available.{RESET_ALL_STYLE}")
             return None

        # --- Validate and Format Position Size ---
        try:
            if not (isinstance(position_size, Decimal) and position_size > Decimal('0')):
                self.logger.error(
                    f"{NEON_RED}Trade aborted for {symbol} ({side}): Invalid position_size ({position_size}). Must be a positive Decimal.{RESET_ALL_STYLE}"
                )
                return None

            # Format amount to exchange's precision rules using market_info
            # Use amount_to_precision with Decimal for better control
            amount_precision_places = market_info.get('amountPrecision', 8) # Default if not found
            try:
                 amount_str_for_api = self.exchange.decimal_to_precision(
                     position_size, ccxt_async.ROUND, amount_precision_places,
                     ccxt_async.DECIMAL_PLACES, ccxt_async.NO_PADDING # Use NO_PADDING for cleaner strings
                 )
                 # Ensure the formatted string can be safely converted back to Decimal or float and is positive
                 amount_for_api_dec = Decimal(amount_str_for_api)
                 if amount_for_api_dec <= Decimal('0'):
                      self.logger.error(
                          f"{NEON_RED}Trade aborted for {symbol} ({side}): Position size after formatting ({amount_str_for_api}) is not positive.{RESET_ALL_STYLE}"
                      )
                      return None
                 amount_for_api_float = float(amount_for_api_dec) # CCXT create_order typically expects float amount

            except Exception as e:
                self.logger.error(
                    f"{NEON_RED}Trade aborted for {symbol} ({side}): Error formatting position_size {position_size} for API: {e}{RESET_ALL_STYLE}", exc_info=True
                )
                return None

        except Exception as e: # Catch errors during initial size validation/formatting
            self.logger.error(
                 f"{NEON_RED}Trade aborted for {symbol} ({side}): Unexpected error during size validation/formatting: {e}{RESET_ALL_STYLE}", exc_info=True
            )
            return None


        # --- Validate and Format Limit Price (if applicable) ---
        price_for_api: Optional[float] = None
        price_log_str: Optional[str] = None # For logging purposes
        if order_type.lower() == 'limit':
            try:
                if not (isinstance(limit_price, Decimal) and limit_price > Decimal('0')):
                    self.logger.error(
                        f"{NEON_RED}Trade aborted for {symbol} ({side}): Limit order chosen, but invalid limit_price ({limit_price}). Must be a positive Decimal.{RESET_ALL_STYLE}"
                    )
                    return None
                try:
                    # Format price to exchange's precision rules using market_info
                    price_log_str = self.exchange.price_to_precision(symbol, float(limit_price))
                    price_for_api = float(price_log_str) # CCXT create_order typically expects float price
                    if price_for_api <= 0:
                        raise ValueError("Formatted limit price is not positive.")
                except Exception as e:
                    self.logger.error(
                        f"{NEON_RED}Trade aborted for {symbol} ({side}): Error formatting limit_price {limit_price} for API: {e}{RESET_ALL_STYLE}", exc_info=True
                    )
                    return None
            except Exception as e: # Catch errors during limit price validation/formatting
                 self.logger.error(
                      f"{NEON_RED}Trade aborted for {symbol} ({side}): Unexpected error during limit price validation/formatting: {e}{RESET_ALL_STYLE}", exc_info=True
                 )
                 return None

        elif order_type.lower() != 'market':
            self.logger.error(f"{NEON_RED}Unsupported order type '{order_type}' for {symbol}. Only 'market' or 'limit' supported.{RESET_ALL_STYLE}")
            return None

        # --- Prepare Parameters for the Order ---
        # Prioritize createOrderParams from exchange options, then add Bybit V5 defaults
        final_params = self.exchange.safe_value(self.exchange.options, 'createOrderParams', {}).copy()

        # Add common parameters
        final_params['reduceOnly'] = reduce_only # reduceOnly is explicit

        # Bybit V5 specific: positionIdx and category
        if self.exchange.id == 'bybit':
            # positionIdx=0 for One-Way mode.
            # For Hedge Mode, 1 for Buy side, 2 for Sell side. This needs more context if hedge mode is used.
            # Assuming One-Way mode (positionIdx=0) as default unless specified otherwise in options/params.
            if 'positionIdx' not in final_params:
                 final_params['positionIdx'] = 0
                 self.logger.debug("Setting default positionIdx=0 for Bybit order.")

            # Ensure Bybit V5 category is included
            if 'category' not in final_params:
                 market_type = market_info.get('type', '').lower()
                 if market_type in ['linear', 'inverse', 'spot']:
                     final_params['category'] = market_type
                 elif market_info.get('linear', False):
                     final_params['category'] = 'linear'
                 elif market_info.get('inverse', False):
                     final_params['category'] = 'inverse'
                 elif self.exchange.options.get('defaultType') == 'unified':
                     # For unified, create order needs linear/inverse/spot category
                     final_params['category'] = 'linear' # Default to linear
                     self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit V5 category for createOrder from market info type '{market_type}' or defaultType 'unified'. Defaulting to 'linear'.{RESET_ALL_STYLE}")
                 else:
                     self.logger.warning(f"{NEON_YELLOW}Could not determine Bybit V5 category for createOrder for {symbol}. Proceeding without category param.{RESET_ALL_STYLE}")

            # Add recvWindow if not present in params or options
            if 'recvWindow' not in final_params and 'recvWindow' not in self.exchange.options.get('options', {}):
                final_params['recvWindow'] = 5000 # milliseconds
                self.logger.debug("Setting recvWindow param for Bybit createOrder.")


        # For market reduce-only orders, IOC (Immediate Or Cancel) is often preferred/required
        if reduce_only and order_type.lower() == 'market' and 'timeInForce' not in final_params:
            final_params['timeInForce'] = 'IOC'
            self.logger.debug("Setting timeInForce=IOC for market reduce-only order.")

        if params: # Merge any additional user-supplied params, user params override defaults
            final_params.update(params)

        # Log the parameters being sent, masking sensitive ones if any explicitly added
        log_params = final_params.copy()
        # Mask sensitive fields if they appear in params (e.g., clientOrderId if it contains sensitive info)
        # if 'clientOrderId' in log_params: log_params['clientOrderId'] = '***'


        base_currency = market_info.get('base', 'units')
        log_message = (
            f"Placing {action_description} {side.upper()} {order_type.upper()} order for {symbol}: "
            f"Size = {amount_for_api_dec} {base_currency}" # Log Decimal amount for clarity
        )
        if price_log_str:
            log_message += f", Price = {price_log_str}"
        log_message += f", Params = {log_params}" # Log masked params
        self.logger.info(log_message)

        # --- Place the Order ---
        try:
            # CCXT create_order typically expects float for amount and price (or None for market price)
            order = await self.exchange.create_order(
                symbol, order_type.lower(), side, amount_for_api_float, price_for_api, final_params
            )
            if order:
                self.logger.info(
                    f"{NEON_GREEN}{action_description} order for {symbol} PLACED successfully. "
                    f"ID: {order.get('id')}, Status: {order.get('status', 'N/A')}{RESET_ALL_STYLE}"
                )
                return order
            else:
                # This case (order is None without exception) should be rare with CCXT
                self.logger.error(f"{NEON_RED}Order placement for {symbol} returned None without raising an exception.{RESET_ALL_STYLE}")
                return None
        except ccxt_async.InsufficientFunds as e:
            self.logger.error(f"{NEON_RED}Insufficient funds to place {side} {order_type} order for {symbol}: {e}{RESET_ALL_STYLE}")
        except ccxt_async.InvalidOrder as e:
            self.logger.error(
                f"{NEON_RED}Invalid order parameters for {symbol} ({side}, {order_type}): {e}. "
                f"Details: Amount={amount_for_api_dec}, Price={limit_price}, Params={log_params}{RESET_ALL_STYLE}", # Log Decimal amount/price, masked params
                exc_info=True
            )
        except ccxt_async.ExchangeError as e: # Broader exchange errors during order placement
            self.logger.error(
                f"{NEON_RED}Exchange error placing {action_description} order for {symbol}: {e}{RESET_ALL_STYLE}", exc_info=True
            )
        except Exception as e: # Other unexpected errors
            self.logger.error(
                f"{NEON_RED}Unexpected error placing {action_description} order for {symbol}: {e}{RESET_ALL_STYLE}", exc_info=True
            )
        return None


    async def _set_position_protection(
        self, symbol: str, market_info: Dict, position_info: Dict,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        trailing_stop_distance: Optional[Union[Decimal, str]] = None, # Can be Decimal distance or "0" string to remove
        tsl_activation_price: Optional[Union[Decimal, str]] = None # Can be Decimal price or "0" string for immediate
    ) -> bool:
        """
        Sets/updates Stop Loss, Take Profit, and Trailing Stop for a position (Bybit V5 specific).

        This is a low-level helper function primarily for Bybit V5's SetTradingStop endpoint.
        It handles parameter formatting and API calls.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            market_info: The market information dictionary for the symbol.
            position_info: The standardized position information dictionary.
            stop_loss_price: The desired Stop Loss price as Decimal, Decimal('0') to remove, or None to leave unchanged.
            take_profit_price: The desired Take Profit price as Decimal, Decimal('0') to remove, or None to leave unchanged.
            trailing_stop_distance: The desired Trailing Stop distance as Decimal, "0" string to remove, or None to leave unchanged.
            tsl_activation_price: The desired Trailing Stop activation price as Decimal, "0" string for immediate activation, or None to leave unchanged.

        Returns:
            True if the protection settings were successfully sent/updated, False otherwise.
        """
        # This logic is highly specific to Bybit V5 API's SetTradingStop endpoint
        # Consider making this more generic or clearly separating exchange-specific logic
        if 'bybit' not in self.exchange.id.lower():
            self.logger.error("Position protection logic (_set_position_protection) is currently Bybit V5 specific.")
            return False
        if not market_info.get('is_contract', False):
            self.logger.warning(f"Protection setting skipped for {symbol}: not a contract market.")
            return True # No action needed if not a contract

        if not position_info or 'side' not in position_info or not position_info['side']:
            self.logger.error(f"{NEON_RED}Cannot set protection for {symbol}: invalid or missing position_info (especially 'side').{RESET_ALL_STYLE}")
            return False

        pos_side_str = position_info['side'].lower() # 'long' or 'short'
        # Bybit V5 positionIdx: 0 for one-way, 1 for buy (long) in hedge, 2 for sell (short) in hedge
        # Get positionIdx from position_info if available, default to 0 (one-way)
        pos_idx_raw = self.exchange.safe_value(position_info, ['info', 'positionIdx'], 0)
        try:
            position_idx = int(pos_idx_raw)
        except (ValueError, TypeError):
            self.logger.warning(f"{NEON_YELLOW}Invalid positionIdx '{pos_idx_raw}' for {symbol}, defaulting to 0.{RESET_ALL_STYLE}")
            position_idx = 0

        # Determine category (linear/inverse/unified) for Bybit V5 API based on market info
        # Prioritize market type, then defaultType
        market_type = market_info.get('type', '').lower()
        category: Optional[str] = None
        if market_type in ['linear', 'inverse', 'spot']:
            category = market_type
        elif market_info.get('linear', False): # Older flags
            category = 'linear'
        elif market_info.get('inverse', False): # Older flags
             category = 'inverse'
        elif self.exchange.options.get('defaultType') == 'unified': # Use default type from exchange options
            # For SetTradingStop, unified account needs linear/inverse/spot category based on symbol type
            # Infer from symbol's quote currency (USDT/USDC -> linear, BTC -> inverse)
            quote = market_info.get('quote', '').upper()
            if quote in ['USDT', 'USDC']:
                 category = 'linear'
            elif quote == 'BTC':
                 category = 'inverse'
            elif market_type in ['swap', 'future', 'option']: # If market type is specific, use it
                 category = market_type
            else:
                 # Default to linear if inference fails for unified
                 category = 'linear'
                 self.logger.warning(f"{NEON_YELLOW}Could not infer Bybit V5 category for SetTradingStop from market info type '{market_type}', quote '{quote}', or defaultType 'unified'. Defaulting to 'linear'.{RESET_ALL_STYLE}")
        else: # Fallback, try 'linear' if type is ambiguous (e.g. 'swap')
            category = 'linear'
            self.logger.warning(f"{NEON_YELLOW}Market category for {symbol} is ambiguous (type: {market_type}, defaultType: {self.exchange.options.get('defaultType')}). Defaulting to 'linear' for protection API call.{RESET_ALL_STYLE}")

        if not category:
            self.logger.error(f"{NEON_RED}Could not determine valid category for setting protection on {symbol}.{RESET_ALL_STYLE}")
            return False

        # Get default protection params from config/exchange options
        # Prioritize setTradingStopParams from exchange options
        api_params: Dict[str, Any] = self.exchange.safe_value(self.exchange.options, 'setTradingStopParams', {}).copy()

        # Explicitly set required params for Bybit V5
        api_params['category'] = category
        api_params['symbol'] = market_info['id'] # Exchange-specific symbol ID
        api_params['positionIdx'] = position_idx # Required for V5

        # Add recvWindow if not present in params or options
        if 'recvWindow' not in api_params and 'recvWindow' not in self.exchange.options.get('options', {}):
            api_params['recvWindow'] = 5000 # milliseconds
            self.logger.debug("Setting recvWindow param for Bybit SetTradingStop.")


        log_parts = [
            f"Attempting to set/update protection for {symbol} ({pos_side_str.upper()}, PosIdx:{position_idx}, Cat:{category}):"
        ]
        protection_fields_to_send: Dict[str, str] = {} # Fields like stopLoss, takeProfit, trailingStop, activePrice

        try:
            # Get price precision and tick size for formatting
            price_precision_places = get_price_precision(market_info, self.logger) # Number of decimal places for price
            min_tick_size_dec = get_min_tick_size(market_info, self.logger)
            if not (min_tick_size_dec and min_tick_size_dec > Decimal('0')): # Fallback if min_tick_size is invalid
                min_tick_size_dec = Decimal(f'1e-{price_precision_places}')
                self.logger.warning(f"{NEON_YELLOW}Using derived tick size {min_tick_size_dec} based on price precision for formatting.{RESET_ALL_STYLE}")

            if not (min_tick_size_dec > Decimal('0')): # Final safety check
                 self.logger.error(f"{NEON_RED}Could not determine a valid positive min_tick_size for formatting protection prices for {symbol}.{RESET_ALL_STYLE}")
                 return False

            def format_price_for_api(price_decimal: Optional[Decimal]) -> Optional[str]:
                """Helper to format a Decimal price for API, handling None and non-positive."""
                if not (price_decimal and isinstance(price_decimal, Decimal)):
                    return None # Invalid input (None, non-Decimal)
                if price_decimal <= Decimal('0'):
                     # For price formatting, we only handle strictly positive values.
                     # Sending "0" to remove is handled separately.
                     self.logger.warning(f"Attempted to format non-positive price {price_decimal} using format_price_for_api for {symbol}.")
                     return None
                try:
                    # Use CCXT's price_to_precision for correct formatting per exchange rules
                    # Ensure the result is a string
                    return str(self.exchange.price_to_precision(symbol, float(price_decimal)))
                except Exception as fmt_e:
                    self.logger.error(f"{NEON_RED}Error using CCXT price_to_precision for {price_decimal} on {symbol}: {fmt_e}{RESET_ALL_STYLE}", exc_info=True)
                    return None

            # Trailing Stop (TSL)
            # Bybit: 'trailingStop' is the distance value. 'activePrice' is the trigger price.
            # 'trailing_stop_distance' can be Decimal (for value) or "0" (string, to remove TSL).
            # 'tsl_activation_price' can be Decimal (for value) or "0" (string, for immediate activation).
            if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > Decimal('0'):
                # Calculate precision for distance (usually same as price precision for Bybit V5)
                # Derive distance precision from tick size
                distance_precision_places = abs(min_tick_size_dec.normalize().as_tuple().exponent)
                try:
                    # Use CCXT's decimal_to_precision for the distance value
                    tsl_dist_str = self.exchange.decimal_to_precision(
                        trailing_stop_distance, ccxt_async.ROUND, distance_precision_places,
                        ccxt_async.DECIMAL_PLACES, ccxt_async.NO_PADDING
                    )
                     # Ensure formatted distance is at least one tick
                    if Decimal(tsl_dist_str) < min_tick_size_dec:
                        # Quantize up to the tick size's precision
                        tsl_dist_str = str(min_tick_size_dec.quantize(Decimal(f'1e-{distance_precision_places}'), rounding=ROUND_UP))
                        self.logger.debug(f"Adjusted TSL distance to minimum tick size for {symbol}: {tsl_dist_str}")

                    # Ensure the formatted distance is still positive
                    if Decimal(tsl_dist_str) <= Decimal('0'):
                         raise ValueError(f"Formatted TSL distance is non-positive: {tsl_dist_str}")

                except Exception as fmt_d_e:
                     self.logger.error(f"{NEON_RED}Error formatting TSL distance {trailing_stop_distance} for {symbol}: {fmt_d_e}{RESET_ALL_STYLE}", exc_info=True)
                     tsl_dist_str = None # Indicate formatting failed

                tsl_act_price_str_final: Optional[str] = None
                if isinstance(tsl_activation_price, str) and tsl_activation_price == "0": # Special string "0" for immediate activation
                    tsl_act_price_str_final = "0"
                elif isinstance(tsl_activation_price, Decimal) and tsl_activation_price > Decimal('0'):
                    tsl_act_price_str_final = format_price_for_api(tsl_activation_price) # Use helper function
                # If tsl_activation_price is None or <= 0 Decimal, tsl_act_price_str_final remains None

                if tsl_dist_str is not None and tsl_act_price_str_final is not None:
                    protection_fields_to_send.update({'trailingStop': tsl_dist_str, 'activePrice': tsl_act_price_str_final})
                    log_parts.append(f"  - Trailing Stop: Distance={tsl_dist_str}, ActivationPrice={tsl_act_price_str_final}")
                else:
                    self.logger.error(
                        f"{NEON_RED}Failed to format TSL parameters for {symbol}. "
                        f"DistanceInput='{trailing_stop_distance}', FormattedDist='{tsl_dist_str}', "
                        f"ActivationInput='{tsl_activation_price}', FormattedAct='{tsl_act_price_str_final}'{RESET_ALL_STYLE}"
                    )
            elif isinstance(trailing_stop_distance, str) and trailing_stop_distance == "0": # Explicitly remove TSL by sending "0" string distance
                protection_fields_to_send['trailingStop'] = "0"
                # When removing TSL, activePrice might also need to be "0" or omitted depending on API.
                # Bybit docs: "To cancel the TS, set trailingStop to '0'." activePrice seems not needed then.
                # If activePrice was in api_params from a previous logic path, ensure it's handled if needed.
                # If the user explicitly passed tsl_activation_price="0" along with trailing_stop_distance="0",
                # we should include activePrice: "0" as well.
                if isinstance(tsl_activation_price, str) and tsl_activation_price == "0":
                     protection_fields_to_send['activePrice'] = "0"
                     log_parts.append("  - Trailing Stop: Removing (distance and activation price set to '0')")
                else:
                     log_parts.append("  - Trailing Stop: Removing (distance set to '0')")

            # If trailing_stop_distance is None or invalid Decimal <= 0, do nothing related to TSL


            # Fixed Stop Loss
            # Allow Decimal(0) or None to mean remove SL or leave unchanged
            if stop_loss_price is not None: # Check if user provided a value (including Decimal('0'))
                if isinstance(stop_loss_price, Decimal) and stop_loss_price > Decimal('0'):
                    sl_price_str = format_price_for_api(stop_loss_price)
                    if sl_price_str is not None: # format_price_for_api returns None for invalid input
                        protection_fields_to_send['stopLoss'] = sl_price_str
                        log_parts.append(f"  - Fixed Stop Loss: {sl_price_str}")
                elif isinstance(stop_loss_price, Decimal) and stop_loss_price == Decimal('0'):
                    protection_fields_to_send['stopLoss'] = "0" # Send "0" string to remove SL
                    log_parts.append(f"  - Fixed Stop Loss: Removing (price set to 0)")
                else: # stop_loss_price is Decimal <= 0 but not Decimal('0') or not a Decimal
                     self.logger.warning(f"{NEON_YELLOW}Invalid stop_loss_price value '{stop_loss_price}' for {symbol}. Not setting fixed SL.{RESET_ALL_STYLE}")


            # Fixed Take Profit
            # Allow Decimal(0) or None to mean remove TP or leave unchanged
            if take_profit_price is not None: # Check if user provided a value (including Decimal('0'))
                if isinstance(take_profit_price, Decimal) and take_profit_price > Decimal('0'):
                     tp_price_str = format_price_for_api(take_profit_price)
                     if tp_price_str is not None:
                         protection_fields_to_send['takeProfit'] = tp_price_str
                         log_parts.append(f"  - Fixed Take Profit: {tp_price_str}")
                elif isinstance(take_profit_price, Decimal) and take_profit_price == Decimal('0'):
                     protection_fields_to_send['takeProfit'] = "0" # Send "0" string to remove TP
                     log_parts.append(f"  - Fixed Take Profit: Removing (price set to 0)")
                else: # take_profit_price is Decimal <= 0 but not Decimal('0') or not a Decimal
                     self.logger.warning(f"{NEON_YELLOW}Invalid take_profit_price value '{take_profit_price}' for {symbol}. Not setting fixed TP.{RESET_ALL_STYLE}")


        except Exception as fmt_err:
            self.logger.error(f"{NEON_RED}Error formatting protection parameters for {symbol}: {fmt_err}{RESET_ALL_STYLE}", exc_info=True)
            return False

        if not protection_fields_to_send:
            self.logger.info(f"No valid protection parameters to set or update for {symbol}.")
            return True # Nothing to do, considered success

        # Add the formatted protection fields to the API parameters
        api_params.update(protection_fields_to_send)

        # Log the action being taken
        self.logger.info("\n".join(log_parts))

        # Log the parameters being sent, masking sensitive ones if any
        log_params = api_params.copy()
        # Mask sensitive fields if they exist (unlikely for trading stop params, but good practice)
        # if 'api_key' in log_params: log_params['api_key'] = '***'

        self.logger.debug(f"  API Call to set trading stop/protection for {symbol}: params={log_params}")

        # --- API Call to set protection ---
        # Bybit V5 endpoint for SL/TP/TSL is typically v5PrivatePostPositionSetTradingStop
        method_name_camel = "v5PrivatePostPositionSetTradingStop"

        if not hasattr(self.exchange, method_name_camel):
            self.logger.error(
                f"{NEON_RED}CCXT instance for {self.exchange.id} is missing the required method for setting position protection "
                f"(checked for '{method_name_camel}'). "
                f"Ensure CCXT library is up-to-date and supports Bybit V5 position protection.{RESET_ALL_STYLE}"
            )
            return False

        set_protection_method = getattr(self.exchange, method_name_camel)

        try:
            # CCXT expects params dict directly for custom API calls
            response = await set_protection_method(api_params)
            self.logger.debug(f"Set protection raw API response for {symbol}: {response}")

            # Check Bybit V5 response structure (retCode = 0 for success)
            if isinstance(response, dict) and response.get('retCode') == 0:
                self.logger.info(
                    f"{NEON_GREEN}Protection for {symbol} successfully set/updated. "
                    f"Message: {response.get('retMsg', 'OK')}{RESET_ALL_STYLE}"
                )
                return True
            else:
                self.logger.error(
                    f"{NEON_RED}Failed to set protection for {symbol}. "
                    f"API Response Code: {response.get('retCode')}, Message: {response.get('retMsg')}{RESET_ALL_STYLE}"
                )
                return False
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error during API call to set protection for {symbol}: {e}{RESET_ALL_STYLE}", exc_info=True)
            return False


    async def set_trailing_stop_loss(
        self, symbol: str, position_info: Dict, config: Dict[str, Any]
    ) -> bool:
        """
        Configures Trailing Stop Loss for an open position based on config settings.

        Also allows setting a Take Profit price simultaneously if included in config.
        This function calculates the TSL parameters and calls the Bybit-specific
        _set_position_protection helper.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT").
            position_info: The standardized open position information dictionary.
            config: The application configuration dictionary.

        Returns:
            True if the TSL (and optional TP) setting was successful or skipped, False otherwise.
        """
        # Check if TSL is globally enabled in config
        enable_tsl = config.get("enable_trailing_stop", False)
        # Check if TP is enabled and configured to be set with TSL
        enable_tp_with_tsl = config.get("enable_take_profit", False) and config.get("set_tp_with_tsl", False)

        # Get TP price from config if enabled to be set with TSL
        take_profit_price: Optional[Decimal] = None
        if enable_tp_with_tsl:
            tp_price_raw = config.get("take_profit_price") # Assuming TP price is in config
            if tp_price_raw is not None:
                try:
                    take_profit_price = Decimal(str(tp_price_raw))
                    if take_profit_price <= Decimal('0'): # If TP is 0 or negative, treat as request to remove
                         self.logger.info(f"Configured Take Profit price for {symbol} is <= 0 ({take_profit_price}), will attempt to remove existing TP.")
                         take_profit_price = Decimal('0') # Explicitly set to 0 Decimal to signal removal
                    else:
                         self.logger.debug(f"Using configured Take Profit price for {symbol}: {take_profit_price}")
                except (InvalidOperation, TypeError):
                    self.logger.warning(f"{NEON_YELLOW}Invalid Take Profit price in config for {symbol}: '{tp_price_raw}'. Not setting/removing TP with TSL.{RESET_ALL_STYLE}")
                    take_profit_price = None # Invalid config value, don't attempt to set TP

        if not enable_tsl:
            self.logger.info(f"Trailing Stop Loss is disabled in config for {symbol}.")
            # If TSL disabled but TP is provided (and valid), still try to set/remove TP
            if take_profit_price is not None:
                self.logger.info(f"TSL disabled, but attempting to set/remove provided Take Profit for {symbol}.")
                market_info = await self.get_market_info(symbol) # Fetch market_info here
                if not market_info:
                     self.logger.error(f"{NEON_RED}Cannot set/remove TP for {symbol}: Market info not available.{RESET_ALL_STYLE}")
                     return False
                return await self._set_position_protection(
                    symbol, market_info, position_info, # Pass market_info
                    stop_loss_price=None, # Don't set SL here
                    take_profit_price=take_profit_price # Pass the provided TP value (Decimal or Decimal('0'))
                )
            # If TP is also None, and TSL is disabled, then no protection needs setting here.
            return True # No TSL action needed, considered success in this context

        market_info = await self.get_market_info(symbol) # Use instance method
        if not market_info:
             self.logger.error(f"{NEON_RED}Cannot set TSL for {symbol}: Market info not available.{RESET_ALL_STYLE}")
             return False

        # Check if the market is a contract market (TSL is typically for contracts)
        if not market_info.get('is_contract', False):
            self.logger.warning(f"{NEON_YELLOW}Trailing Stop Loss is typically for contract markets. Skipped for {symbol} (not a contract).{RESET_ALL_STYLE}")
            # If TP is provided for a non-contract, still try to set it if API allows (e.g. spot TP orders)
            if take_profit_price is not None: # Check if user provided a value for TP
                self.logger.info(f"Market is not a contract, but attempting to set/remove provided Take Profit for {symbol}.")
                return await self._set_position_protection(
                    symbol, market_info, position_info, self.logger,
                    stop_loss_price=None, # Don't set SL here
                    take_profit_price=take_profit_price # Pass the provided TP value (Decimal or Decimal('0'))
                )
            return True # No TSL action needed, considered success

        # Load TSL parameters from config
        try:
            callback_rate_str = str(config.get("trailing_stop_callback_rate", "0.005")) # e.g., 0.5%
            activation_percentage_str = str(config.get("trailing_stop_activation_percentage", "0.003")) # e.g., 0.3%

            callback_rate = Decimal(callback_rate_str)
            activation_percentage = Decimal(activation_percentage_str)

            if callback_rate <= Decimal('0'):
                raise ValueError("Trailing stop callback rate must be positive.")
            if activation_percentage < Decimal('0'): # Allow 0 for activation at entry or immediate
                raise ValueError("Trailing stop activation percentage must be non-negative.")
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Invalid TSL parameters in configuration for {symbol}: {e}. Please check config.{RESET_ALL_STYLE}")
            return False

        # Get required position info
        try:
            entry_price = position_info.get('entryPriceDecimal')
            position_side = position_info.get('side', '').lower() # 'long' or 'short'
            if not (isinstance(entry_price, Decimal) and entry_price > Decimal('0')):
                raise ValueError(f"Invalid or missing entry price in position_info: {entry_price}")
            if position_side not in ['long', 'short']:
                raise ValueError(f"Invalid or missing side in position_info: {position_side}")
        except (ValueError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Invalid position information for TSL setup ({symbol}): {e}. Position: {position_info}{RESET_ALL_STYLE}")
            return False

        try:
            # Get price precision and tick size for calculations and formatting
            price_precision_places = get_price_precision(market_info, self.logger)
            min_tick_size = get_min_tick_size(market_info, self.logger)
            # Fallback tick size if not properly defined in market_info
            quantize_fallback_tick = Decimal(f'1e-{price_precision_places}')
            effective_tick_size = min_tick_size if min_tick_size and min_tick_size > Decimal('0') else quantize_fallback_tick
            if not (effective_tick_size > Decimal('0')): # Should not happen with fallback
                self.logger.error(f"{NEON_RED}Could not determine a valid tick size for TSL calculations for {symbol}.{RESET_ALL_STYLE}")
                return False

            # Fetch current market price to aid activation logic, fallback to entry price if fetch fails
            current_market_price = await self.fetch_current_price(symbol) # Use instance method
            if not (current_market_price and isinstance(current_market_price, Decimal) and current_market_price > Decimal('0')):
                self.logger.warning(f"{NEON_YELLOW}Could not fetch valid current market price for {symbol} for TSL logic ({current_market_price}). Using entry price ({entry_price}) as reference.{RESET_ALL_STYLE}")
                current_market_price = entry_price # Use entry price as the reference point

            # Calculate theoretical activation price based on entry price and activation percentage
            price_change_for_activation = entry_price * activation_percentage
            # Raw activation price based on entry and percentage
            raw_activation_price = entry_price + (price_change_for_activation if position_side == 'long' else -price_change_for_activation)

            # Determine if position is already profitable enough for TSL to activate immediately (Bybit: activePrice="0")
            # This check compares the current market price to the theoretical activation price
            activate_immediately = False
            if config.get("tsl_activate_immediately_if_profitable", True):
                # Check if current market price is *at or past* the calculated raw activation price in the profitable direction
                if position_side == 'long' and current_market_price >= raw_activation_price:
                    activate_immediately = True
                elif position_side == 'short' and current_market_price <= raw_activation_price:
                    activate_immediately = True

            final_activation_price_param: Union[Decimal, str] # This will be passed to _set_position_protection
            calculated_activation_price_for_log: Optional[Decimal] = None # For logging clarity

            if activate_immediately:
                final_activation_price_param = "0" # Bybit: "0" for activePrice means activate immediately
                # For logging, show the current price that triggered immediate activation
                calculated_activation_price_for_log = current_market_price
                self.logger.info(
                    f"TSL for {symbol} ({position_side}): Position is already profitable beyond activation point ({raw_activation_price:.{price_precision_places}f} calculated). "
                    f"Setting activePrice='0' for immediate trailing based on current market price ({current_market_price:.{price_precision_places}f})."
                )
            else:
                # If not activating immediately, calculate the specific activation price to set.
                # Ensure it's at least one tick into profit from entry and respects market tick size.
                min_profit_activation_price = entry_price + effective_tick_size if position_side == 'long' else entry_price - effective_tick_size

                # The activation price should be the maximum (for long) or minimum (for short) of:
                # 1. The calculated raw activation price based on percentage.
                # 2. The current market price (if it's already past the raw activation price, but not enough for immediate activation).
                # 3. One tick into profit from the entry price (minimum profitable level).

                calculated_activation_price = raw_activation_price # Start with the percentage-based price

                # Ensure activation price is at least one tick into profit
                if position_side == 'long' and calculated_activation_price < min_profit_activation_price:
                     calculated_activation_price = min_profit_activation_price
                elif position_side == 'short' and calculated_activation_price > min_profit_activation_price:
                     calculated_activation_price = min_profit_activation_price

                # Ensure activation price is ahead of current market price if not activating immediately
                # (This handles cases where current price is between entry and calculated activation,
                # or slightly past calculated activation but not enough for the 'activate_immediately' threshold)
                if position_side == 'long' and calculated_activation_price < current_market_price:
                     calculated_activation_price = current_market_price
                elif position_side == 'short' and calculated_activation_price > current_market_price:
                     calculated_activation_price = current_market_price

                # Quantize the final calculated activation price to the nearest tick size
                # Rounding depends on side to ensure the activation price is set correctly relative to profitable movement
                rounding_mode = ROUND_DOWN if position_side == 'long' else ROUND_UP # For long, round down to activate sooner; for short, round up
                # Quantize to the precision of the tick size
                calculated_activation_price = (calculated_activation_price / effective_tick_size).quantize(Decimal('1'), rounding=rounding_mode) * effective_tick_size

                # Final validation of calculated activation price - must be positive and profitable
                if calculated_activation_price <= Decimal('0'):
                    self.logger.error(f"{NEON_RED}Final calculated TSL Activation Price ({calculated_activation_price}) is not positive for {symbol}. Cannot set TSL.{RESET_ALL_STYLE}")
                    return False
                # Re-check profitability after quantization - ensure it's strictly better than entry
                if position_side == 'long' and calculated_activation_price <= entry_price:
                     self.logger.warning(
                         f"{NEON_YELLOW}Final calculated TSL Activation Price ({calculated_activation_price}) for LONG {symbol} is not strictly profitable vs Entry ({entry_price}). "
                         f"Adjusting to one tick above entry after quantization.{RESET_ALL_STYLE}"
                     )
                     # Calculate one tick above entry and quantize
                     calculated_activation_price = ((entry_price + effective_tick_size) / effective_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * effective_tick_size
                elif position_side == 'short' and calculated_activation_price >= entry_price:
                     self.logger.warning(
                         f"{NEON_YELLOW}Final calculated TSL Activation Price ({calculated_activation_price}) for SHORT {symbol} is not strictly profitable vs Entry ({entry_price}). "
                         f"Adjusting to one tick below entry after quantization.{RESET_ALL_STYLE}"
                     )
                     # Calculate one tick below entry and quantize
                     calculated_activation_price = ((entry_price - effective_tick_size) / effective_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * effective_tick_size

                if calculated_activation_price <= Decimal('0'): # Re-check after adjustment
                     self.logger.error(f"{NEON_RED}Final adjusted TSL Activation Price ({calculated_activation_price}) non-positive for {symbol}. Cannot set TSL.{RESET_ALL_STYLE}"); return False


                final_activation_price_param = calculated_activation_price
                calculated_activation_price_for_log = calculated_activation_price


            # Calculate trailing distance based on callback rate and entry price (or current price, depending on strategy)
            # Here, using entry price for calculation base as per original code structure
            raw_trail_distance = entry_price * callback_rate
            # Quantize the trail distance to the nearest tick size
            # Rounding mode for distance: usually round up to be slightly wider/safer
            trail_distance = (raw_trail_distance / effective_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * effective_tick_size
            if trail_distance < effective_tick_size: # Ensure distance is at least one tick
                trail_distance = effective_tick_size
            if trail_distance <= Decimal('0'):
                self.logger.error(f"{NEON_RED}Calculated TSL trail distance ({trail_distance}) is not positive for {symbol}. Cannot set TSL.{RESET_ALL_STYLE}")
                return False

            # Log the calculated TSL parameters before sending to API
            log_act_price_str = f"{calculated_activation_price_for_log:.{price_precision_places}f}" if calculated_activation_price_for_log else "N/A (Immediate)"
            self.logger.info(
                f"Calculated TSL parameters for {symbol} ({position_side.upper()}):\n"
                f"  Entry Price: {entry_price:.{price_precision_places}f}\n"
                f"  Activation Price (for API): '{final_activation_price_param}' (Based on calculated: {log_act_price_str}, from {activation_percentage:.2%})\n"
                f"  Trail Distance: {trail_distance:.{price_precision_places}f} (From callback rate: {callback_rate:.2%})"
            )
            if take_profit_price is not None and isinstance(take_profit_price, Decimal):
                 tp_log_str = f"{take_profit_price:.{price_precision_places}f}" if take_profit_price > Decimal('0') else "Remove"
                 self.logger.info(f"  Also setting Take Profit: {tp_log_str}")
            elif take_profit_price is not None and isinstance(take_profit_price, Decimal) and take_profit_price == Decimal('0'):
                 self.logger.info(f"  Also removing Take Profit as requested.")


            # Call the Bybit-specific helper to set the protection using the calculated parameters
            # Pass take_profit_price as provided by the caller
            return await self._set_position_protection(
                symbol, market_info, position_info, # Pass market_info
                stop_loss_price=None, # TSL typically replaces/manages the stop loss part, don't send fixed SL here
                take_profit_price=take_profit_price, # Pass whatever TP value was provided (Decimal or Decimal('0'))
                trailing_stop_distance=trail_distance, # Pass the calculated Decimal distance
                tsl_activation_price=final_activation_price_param # Pass Decimal or string "0"
            )
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error during TSL setup for {symbol}: {e}{RESET_ALL_STYLE}", exc_info=True)
            return False

    async def close_exchange(self):
        """Closes the exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.logger.info("Exchange connection closed.")

    # --- Placeholder methods from user's original code, adapted to async ---
    # These need proper implementation if you intend to use them.
    # They are included to show where they would fit in the class structure.

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancels an open order."""
        self.logger.warning(f"Cancel order method for {order_id} on {symbol} is a placeholder.")
        # Example implementation:
        # try:
        #     response = await self.exchange.cancel_order(order_id, symbol)
        #     self.logger.info(f"Order {order_id} on {symbol} cancelled: {response}")
        #     return True
        # except Exception as e:
        #     self.logger.error(f"Failed to cancel order {order_id} on {symbol}: {e}")
        #     return False
        raise NotImplementedError("cancel_order method is a placeholder.")

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches information about a specific order."""
        self.logger.warning(f"Fetch order method for {order_id} on {symbol} is a placeholder.")
        # Example implementation:
        # try:
        #     order = await self.exchange.fetch_order(order_id, symbol)
        #     return order
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch order {order_id} on {symbol}: {e}")
        #     return None
        raise NotImplementedError("fetch_order method is a placeholder.")

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches open orders for a symbol or all symbols."""
        self.logger.warning(f"Fetch open orders method for {symbol or 'all symbols'} is a placeholder.")
        # Example implementation:
        # try:
        #     orders = await self.exchange.fetch_open_orders(symbol)
        #     return orders
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch open orders for {symbol}: {e}")
        #     return []
        raise NotImplementedError("fetch_open_orders method is a placeholder.")

    async def fetch_closed_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches closed orders for a symbol or all symbols."""
        self.logger.warning(f"Fetch closed orders method for {symbol or 'all symbols'} is a placeholder.")
        # Example implementation:
        # try:
        #     orders = await self.exchange.fetch_closed_orders(symbol)
        #     return orders
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch closed orders for {symbol}: {e}")
        #     return []
        raise NotImplementedError("fetch_closed_orders method is a placeholder.")

    async def fetch_my_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetches personal trade history for a symbol or all symbols."""
        self.logger.warning(f"Fetch my trades method for {symbol or 'all symbols'} is a placeholder.")
        # Example implementation:
        # try:
        #     trades = await self.exchange.fetch_my_trades(symbol)
        #     return trades
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch my trades for {symbol}: {e}")
        #     return []
        raise NotImplementedError("fetch_my_trades method is a placeholder.")

    async def fetch_total_balance(self) -> Optional[Decimal]:
        """Fetches the total account balance across all currencies."""
        self.logger.warning("Fetch total balance method is a placeholder.")
        # Example implementation:
        # try:
        #     balance_info = await self.exchange.fetch_balance()
        #     total_balance = balance_info.get('total', {}).get('USDT') # Example for USDT
        #     return Decimal(str(total_balance)) if total_balance is not None else None
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch total balance: {e}")
        #     return None
        raise NotImplementedError("fetch_total_balance method is a placeholder.")

    async def fetch_margin_balance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches margin balance information for a symbol."""
        self.logger.warning(f"Fetch margin balance method for {symbol} is a placeholder.")
        # Example implementation (might need specific Bybit V5 endpoint):
        # try:
        #     margin_info = await self.exchange.fetch_margin_balance(symbol) # CCXT might not have a standard method
        #     # You might need to use the manual _make_request or a CCXT custom method call here
        #     # For Bybit V5: v5PrivateGetAccountWalletBalance
        #     # params = {'accountType': 'UNIFIED', 'coin': 'USDT'}
        #     # wallet_balance = await self.exchange.v5PrivateGetAccountWalletBalance(params)
        #     return margin_info # Or parsed wallet_balance
        # except Exception as e:
        #     self.logger.error(f"Failed to fetch margin balance for {symbol}: {e}")
        #     return None
        raise NotImplementedError("fetch_margin_balance method is a placeholder.")


# Example of how to use the class (requires an asyncio event loop)
# async def main():
#     # Assuming config is loaded elsewhere
#     # config = {
#     #     "api_key": "YOUR_API_KEY",
#     #     "api_secret": "YOUR_API_SECRET",
#     #     "use_sandbox": True,
#     #     "exchange_options": {
#     #          "options": {
#     #               "defaultType": "unified"
#     #          }
#     #     }
#     # }
#     # logger = logging.getLogger("main_bot")
#     # logging.basicConfig(level=logging.INFO)

#     # try:
#     #     api_client = BybitAPI(config, logger)
#     #     await api_client.load_markets()
#     #     price = await api_client.fetch_current_price("BTC/USDT")
#     #     if price:
#     #         logger.info(f"Current BTC/USDT price: {price}")
#     #     await api_client.close_exchange()
#     # except Exception as e:
#     #     logger.error(f"An error occurred in main: {e}")

# # To run the example:
# # if __name__ == "__main__":
# #     asyncio.run(main())
