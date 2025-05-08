# File: exchange_api.py
"""
Asynchronous Bybit API client module using CCXT async support within a class structure.

Provides methods for:
- Connecting and initializing the Bybit exchange instance.
- Fetching market data (ticker, klines, orderbook) with retries and validation.
- Managing orders (create, cancel, edit, query, batch ops) and positions.
- Retrieving account balance.
- Setting leverage, position mode, and protection (SL/TP/TSL) using Bybit V5 specifics.
"""

import os
import sys  # Added for stderr in ImportError fallback
import time
import asyncio
import logging
import importlib.metadata  # For getting package version
import random  # For retry jitter
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union  # Callable was already present

import ccxt.async_support as ccxt_async
import pandas as pd

# Import constants and utility functions
try:
    from utils import (
        NEON_GREEN,
        NEON_RED,
        NEON_YELLOW,
        RESET_ALL_STYLE,
        RETRY_DELAY_SECONDS,
        get_min_tick_size,
        get_price_precision,
        _exponential_backoff,
    )
except ImportError:
    print("Error importing from utils in exchange_api.py", file=sys.stderr)
    NEON_GREEN = NEON_RED = NEON_YELLOW = RESET_ALL_STYLE = ""
    RETRY_DELAY_SECONDS = 5.0

    # Fallback definitions matching expected signatures
    def get_price_precision(market: Dict[str, Any], logger: logging.Logger) -> int:
        return 4

    def get_min_tick_size(market: Dict[str, Any], logger: logging.Logger) -> Decimal:
        return Decimal("0.0001")

    def _exponential_backoff(attempt: int, base_delay: float = 5.0, max_cap: float = 60.0) -> float:
        return min(base_delay * (2**attempt), max_cap)


# Module-level logger (can be used for messages before class instance exists or for utility functions)
module_logger = logging.getLogger(__name__)


class BybitAPI:
    """
    Asynchronous Bybit API client using CCXT async support.

    Encapsulates exchange interaction, providing methods for market data,
    trading operations, and account information retrieval with built-in
    retry logic and Bybit V5 parameter handling. Includes caching,
    circuit breaker, and configurable options.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the Bybit API client configuration and CCXT exchange object.

        Args:
            config: Configuration dictionary with keys like 'exchange_id', 'api_key',
                    'api_secret', 'use_sandbox', 'default_market_type', 'quote_currency',
                    'max_api_retries', 'api_timeout_ms', 'market_cache_duration_seconds',
                    'circuit_breaker_cooldown_seconds', 'log_level', 'order_rate_limit_per_second',
                    various default parameter dicts ('exchange_options', 'market_load_params', etc.).
            logger: Logger instance for logging API client activities.

        Raises:
            ValueError: If API keys are missing or the specified exchange_id is invalid.
        """
        self.logger = logger
        self._config = config

        # --- Configure Logging Level ---
        log_level_str = config.get("log_level", "INFO").upper()
        log_level_int = getattr(logging, log_level_str, logging.INFO)
        self.logger.setLevel(log_level_int)
        self.logger.info(f"API Client log level set to: {log_level_str}")

        # --- Credentials ---
        api_key = self._config.get("api_key") or os.environ.get("BYBIT_API_KEY")
        api_secret = self._config.get("api_secret") or os.environ.get("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            self.logger.critical(f"{NEON_RED}API keys not found.{RESET_ALL_STYLE}")
            raise ValueError("API Key and Secret must be provided.")
        self.api_key = api_key
        self.api_secret = api_secret

        # --- Core Config ---
        self.testnet = self._config.get("use_sandbox", False)
        self.exchange_id = self._config.get("exchange_id", "bybit").lower()
        self.quote_currency = self._config.get("quote_currency", "USDT")

        # --- Operational Parameters ---
        self.max_api_retries = self._config.get("max_api_retries", 3)
        self.api_timeout_ms = self._config.get("api_timeout_ms", 15000)
        self.market_cache_duration_seconds = self._config.get("market_cache_duration_seconds", 3600)
        self.order_rate_limit = self._config.get("order_rate_limit_per_second", 10.0)
        self.last_order_time = 0.0

        # --- Circuit Breaker ---
        self.circuit_breaker_cooldown = self._config.get("circuit_breaker_cooldown_seconds", 300.0)
        self.circuit_breaker_tripped = False
        self.circuit_breaker_failure_count = 0
        self.circuit_breaker_max_failures = 5
        self.circuit_breaker_reset_time = 0.0

        if self.exchange_id != "bybit":
            self.logger.warning(
                f"{NEON_YELLOW}Class optimized for 'bybit', but config uses '{self.exchange_id}'. Functionality may vary.{RESET_ALL_STYLE}"
            )

        # --- Initialize CCXT Exchange Object ---
        try:
            if not hasattr(ccxt_async, self.exchange_id):
                raise ValueError(f"Exchange ID '{self.exchange_id}' is not supported by CCXT async.")
            exchange_class = getattr(ccxt_async, self.exchange_id)

            # Consolidate default parameters for CCXT exchange instantiation
            default_options = self._config.get("exchange_options", {}).get("options", {}).copy()
            default_options.setdefault("defaultType", self._config.get("default_market_type", "unified").lower())
            if self.exchange_id == "bybit":
                default_options.setdefault("createOrderRequiresPrice", False)
                default_options.setdefault("recvWindow", 5000)

            exchange_params = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,  # CCXT's internal rate limiter
                "timeout": self.api_timeout_ms,
                "options": default_options,
                # Store default method-specific params directly in exchange.options for ccxt's safe_value access
                # These allow overriding CCXT defaults or providing required params for specific methods
                "loadMarketsParams": self._config.get("market_load_params", {}),
                "balanceFetchParams": self._config.get("balance_fetch_params", {}),
                "fetchPositionsParams": self._config.get("fetch_positions_params", {}),
                "createOrderParams": self._config.get("create_order_params", {}),
                "editOrderParams": self._config.get("edit_order_params", {}),
                "cancelOrderParams": self._config.get("cancel_order_params", {}),
                "cancelAllOrdersParams": self._config.get("cancel_all_orders_params", {}),
                "fetchOrderParams": self._config.get("fetch_order_params", {}),
                "fetchOpenOrdersParams": self._config.get("fetch_open_orders_params", {}),
                "fetchClosedOrdersParams": self._config.get("fetch_closed_orders_params", {}),
                "fetchMyTradesParams": self._config.get("fetch_my_trades_params", {}),
                "setLeverageParams": self._config.get("set_leverage_params", {}),
                "setTradingStopParams": self._config.get("set_trading_stop_params", {}),  # For Bybit V5 specific SL/TP
                "setPositionModeParams": self._config.get(
                    "set_position_mode_params", {}
                ),  # For Bybit V5 specific position mode
            }

            self.exchange: ccxt_async.Exchange = exchange_class(exchange_params)
            self.markets_cache: Dict[str, Any] = {}
            self.last_markets_update_time: float = 0.0
            self.logger.info(
                f"API client configured (ID: {self.exchange.id}, Sandbox: {self.testnet}). Call initialize() to connect and load markets."
            )
        except ValueError as ve:
            self.logger.critical(f"{NEON_RED}Configuration error: {ve}{RESET_ALL_STYLE}")
            raise
        except Exception as e:
            self.logger.critical(
                f"{NEON_RED}Failed to initialize CCXT exchange object: {e}{RESET_ALL_STYLE}", exc_info=True
            )
            raise

    async def initialize(self) -> bool:
        """Completes exchange initialization: set sandbox, load markets, and perform connection checks."""
        try:
            ccxt_version = importlib.metadata.version("ccxt")
            self.logger.info(f"Using CCXT version: {ccxt_version}")
        except importlib.metadata.PackageNotFoundError:
            self.logger.warning("Could not determine CCXT version. 'ccxt' package may not be installed correctly.")
        except Exception as e:  # General exception for other metadata issues
            self.logger.warning(f"Could not get CCXT version due to an unexpected error: {e}")

        try:
            # Set Sandbox Mode if configured
            if self.testnet:
                self.logger.warning(
                    f"{NEON_YELLOW}USING SANDBOX MODE (Testnet) for {self.exchange.id}{RESET_ALL_STYLE}"
                )
                if hasattr(self.exchange, "set_sandbox_mode") and callable(self.exchange.set_sandbox_mode):
                    try:
                        # Await if it's a coroutine, otherwise call directly
                        if asyncio.iscoroutinefunction(self.exchange.set_sandbox_mode):
                            await self.exchange.set_sandbox_mode(True)
                        else:
                            self.exchange.set_sandbox_mode(True)
                        self.logger.info("Sandbox mode enabled via exchange.set_sandbox_mode(True).")
                    except Exception as e:
                        self.logger.warning(
                            f"Call to exchange.set_sandbox_mode(True) failed: {e}. Will attempt manual URL override if applicable."
                        )

                # Manual URL override for Bybit if set_sandbox_mode didn't work or as a primary method
                if self.exchange.id == "bybit":
                    testnet_url = self.exchange.urls.get("test")  # Standard CCXT testnet URL key
                    if not testnet_url:  # Fallback if 'test' key is missing
                        testnet_url = "https://api-testnet.bybit.com"
                        self.logger.info(f"Using default Bybit testnet URL: {testnet_url}")

                    current_api_url = self.exchange.urls.get("api")
                    # Check if already set to testnet (can be complex if 'api' is a dict for different services)
                    is_already_testnet = False
                    if isinstance(current_api_url, dict):
                        is_already_testnet = any(url == testnet_url for url in current_api_url.values())
                    elif isinstance(current_api_url, str):
                        is_already_testnet = current_api_url == testnet_url

                    if not is_already_testnet:
                        self.exchange.urls["api"] = testnet_url
                        self.logger.info(f"Manually set Bybit API URL to testnet: {testnet_url}")
                    else:
                        self.logger.info("Bybit API URL appears to be already configured for testnet.")

            # Load Markets
            if not await self.load_markets(reload=True):  # Force reload during initialization
                raise ccxt_async.ExchangeError("Initial market data load failed.")

            # Connection & Initial Balance Check
            if not await self.check_connection():
                raise ccxt_async.NetworkError("Initial API connection check failed.")

            balance_data = await self.fetch_balance(self.quote_currency)  # Fetch for the primary quote currency
            if balance_data is None:
                self.logger.error(f"{NEON_RED}Initial balance fetch FAILED for {self.quote_currency}.{RESET_ALL_STYLE}")
                # Depending on strictness, one might raise an error here
            elif isinstance(balance_data, Decimal):  # Specific currency balance
                self.logger.info(
                    f"{NEON_GREEN}Initial balance check OK: {balance_data:.4f} {self.quote_currency}{RESET_ALL_STYLE}"
                )
            else:  # Full balance dictionary
                self.logger.info(f"{NEON_GREEN}Initial full balance data fetch OK.{RESET_ALL_STYLE}")

            self.logger.info(
                f"{NEON_GREEN}API Client initialized successfully for {self.exchange.id}.{RESET_ALL_STYLE}"
            )
            return True

        except Exception as e:
            log_msg = f"{NEON_RED}API Client initialization failed: {e}{RESET_ALL_STYLE}"
            if isinstance(e, (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable)):
                # Check for common network-related keywords in the error message
                err_str_lower = str(e).lower()
                if any(keyword in err_str_lower for keyword in ["dns", "resolve", "connect", "timeout"]):
                    log_msg += f"\n{NEON_YELLOW}Hint: This might be a network connectivity issue (DNS, firewall, or internet connection problem).{RESET_ALL_STYLE}"
            elif isinstance(e, ccxt_async.AuthenticationError):
                log_msg += (
                    f"\n{NEON_RED}Hint: Authentication failed. Please check your API key and secret.{RESET_ALL_STYLE}"
                )

            self.logger.critical(log_msg, exc_info=True)
            await self.close()  # Attempt to clean up resources
            return False

    async def close(self):
        """Closes the underlying CCXT exchange connection gracefully."""
        if (
            hasattr(self, "exchange")
            and self.exchange
            and hasattr(self.exchange, "close")
            and callable(self.exchange.close)
        ):
            try:
                await self.exchange.close()
                self.logger.info("Exchange connection closed successfully.")
            except Exception as e:
                self.logger.error(f"Error encountered while closing exchange connection: {e}", exc_info=True)
        else:
            self.logger.info("No active exchange connection to close or close method not available.")

    async def check_connection(self) -> bool:
        """Checks API server connectivity by fetching the server time."""
        try:
            # Fetch server time; a lightweight call to check connectivity and authentication
            server_time_ms = await self.exchange.fetch_time()
            if server_time_ms and server_time_ms > 0:
                server_time_iso = self.exchange.iso8601(server_time_ms)
                self.logger.info(
                    f"{NEON_GREEN}API connection check successful. Server Time: {server_time_iso}{RESET_ALL_STYLE}"
                )

                # If circuit breaker was tripped, reset it upon successful connection
                if self.circuit_breaker_tripped:
                    self.logger.info(
                        f"{NEON_GREEN}Connection re-established. Resetting circuit breaker.{RESET_ALL_STYLE}"
                    )
                    self.circuit_breaker_tripped = False
                    self.circuit_breaker_failure_count = 0
                    self.circuit_breaker_reset_time = 0.0
                return True
            else:
                # This case should ideally be caught by ccxt exceptions if the call fails
                raise ccxt_async.ExchangeError(f"fetch_time returned an invalid or empty response: {server_time_ms}")

        except Exception as e:
            self.logger.error(
                f"{NEON_RED}API connection check FAILED: {e}{RESET_ALL_STYLE}", exc_info=False
            )  # exc_info=False for brevity in repeated checks

            self.circuit_breaker_failure_count += 1
            if (
                not self.circuit_breaker_tripped
                and self.circuit_breaker_failure_count >= self.circuit_breaker_max_failures
            ):
                self.circuit_breaker_tripped = True
                self.circuit_breaker_reset_time = time.monotonic() + self.circuit_breaker_cooldown
                self.logger.critical(
                    f"{NEON_RED}Circuit breaker TRIPPED for {self.circuit_breaker_cooldown:.0f} seconds due to repeated connection failures.{RESET_ALL_STYLE}"
                )
            return False

    async def _handle_fetch_exception(
        self, e: Exception, attempt: int, total_attempts: int, item_desc: str, context_info: Optional[str] = None
    ) -> bool:
        """
        Internal helper to log API request exceptions, determine retry eligibility,
        manage circuit breaker state, and apply appropriate delays.

        Args:
            e: The exception instance.
            attempt: Current retry attempt number (0-indexed).
            total_attempts: Maximum number of attempts.
            item_desc: Description of the item/operation being fetched (e.g., "market data for BTC/USDT").
            context_info: Additional context, like a symbol or ID.

        Returns:
            bool: True if a retry should be attempted, False otherwise.
        """
        # If circuit breaker is tripped and cooldown period is active, do not retry
        if self.circuit_breaker_tripped and time.monotonic() < self.circuit_breaker_reset_time:
            self.logger.error(
                f"{NEON_RED}Circuit breaker ACTIVE. Skipping retry for {item_desc}. Cooldown ends in {self.circuit_breaker_reset_time - time.monotonic():.1f}s.{RESET_ALL_STYLE}"
            )
            return False
        # If circuit breaker was tripped but cooldown has elapsed, reset it
        elif self.circuit_breaker_tripped:
            self.logger.info(
                f"{NEON_YELLOW}Circuit breaker cooldown period elapsed. Resetting circuit breaker.{RESET_ALL_STYLE}"
            )
            self.circuit_breaker_tripped = False
            self.circuit_breaker_failure_count = 0
            self.circuit_breaker_reset_time = 0.0

        is_retryable = False
        delay_seconds = RETRY_DELAY_SECONDS  # Base delay from constants
        exception_details = str(e)
        log_method = self.logger.error  # Default log level for errors

        # Extract exchange-specific error code if available (prefer Bybit's retCode)
        error_code_info = getattr(e, "info", {})
        exchange_specific_code = self.exchange.safe_string(error_code_info, "retCode")  # Bybit V5
        if not exchange_specific_code and isinstance(e, ccxt_async.ExchangeError):  # General CCXT error code
            exchange_specific_code = getattr(e, "code", None)

        # Determine retry eligibility and delay based on exception type
        if isinstance(e, ccxt_async.AuthenticationError):
            message = f"Authentication error while fetching {item_desc}"
            # Typically, auth errors are not retryable unless it's a transient issue.
            # Retry once, as sometimes a temporary glitch or IP restriction change might occur.
            is_retryable = attempt == 0
            delay_seconds = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS * 2)
        elif isinstance(e, (ccxt_async.RateLimitExceeded, ccxt_async.DDoSProtection)):
            log_method = self.logger.warning
            message = f"Rate limit exceeded while fetching {item_desc}"
            is_retryable = True
            # Use 'retry-after' header if present, otherwise exponential backoff
            retry_after_header = self.exchange.safe_integer(getattr(e, "headers", {}), "Retry-After")  # Case-sensitive
            if retry_after_header:
                delay_seconds = float(retry_after_header) + random.uniform(0.1, 1.0)  # Add jitter
            else:
                delay_seconds = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS * 3, max_cap=300.0)
        elif isinstance(
            e,
            (ccxt_async.NetworkError, ccxt_async.RequestTimeout, asyncio.TimeoutError, ccxt_async.ExchangeNotAvailable),
        ):
            log_method = self.logger.warning
            message = f"Network/Timeout error while fetching {item_desc}"
            is_retryable = True
            delay_seconds = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS, max_cap=120.0)
        elif isinstance(e, ccxt_async.ExchangeError):  # General exchange errors
            message = f"Exchange error while fetching {item_desc}"
            err_lower = exception_details.lower()

            if self.exchange_id == "bybit" and exchange_specific_code:
                try:
                    code_int = int(exchange_specific_code)
                    # Bybit V5 specific error codes (non-exhaustive examples)
                    # Codes that indicate a non-retryable client-side or state issue
                    non_retryable_bybit_codes = [
                        10001,  # Parameter error
                        110009,  # Price is too high/low than the liquidation price
                        110045,  # Position status is not normal
                        110013,  # OrderLinkedID is duplicate
                        10003,  # Invalid API key
                        10004,  # Authentication failure / IP not whitelisted
                        130021,  # Insufficient balance
                        110032,  # Risk limit cannot be adjusted due to existing position or active order
                        110017,  # Order quantity exceeds the lower limit
                        110018,  # Order quantity exceeds the upper limit
                        # Add more as identified...
                    ]
                    # Codes that might indicate a temporary server-side issue or state that might change
                    retryable_bybit_codes = [
                        10002,  # System busy / Server error
                        10006,  # Too many visits / Rate limit related (though should be caught by RateLimitExceeded)
                        10016,  # Service data exception (can be temporary)
                        30034,  # Order not found (could be due to replication lag if checking immediately after creation)
                        30035,  # No need to cancel order (if already cancelled/filled)
                        10005,  # Permission denied (sometimes transient if permissions are updating)
                        # Standard HTTP error codes often wrapped by Bybit
                        500,
                        502,
                        503,
                        504,
                    ]

                    if (
                        code_int in non_retryable_bybit_codes
                        or "accounttype" in err_lower
                        or "invalid parameter" in err_lower
                    ):
                        message = f"Bybit Non-Retryable Error ({exchange_specific_code}) for {item_desc}"
                        is_retryable = False
                    elif code_int == 110025 and "position" in item_desc.lower():  # "position is not exists"
                        self.logger.info(
                            f"Bybit: Position not found for {context_info or item_desc} (Code: {exchange_specific_code}). Not an error if expecting no position."
                        )
                        return False  # Not an error to retry, but a state (no position)
                    elif code_int in retryable_bybit_codes:
                        log_method = self.logger.warning
                        is_retryable = True
                        message = f"Bybit Temporary Error ({exchange_specific_code}) for {item_desc}"
                        delay_seconds = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS * 2)
                    else:  # Unknown Bybit code, default to retry with caution
                        log_method = self.logger.warning
                        is_retryable = True
                        message = (
                            f"Bybit Exchange Error ({exchange_specific_code}) for {item_desc} (retrying as precaution)"
                        )
                        delay_seconds = _exponential_backoff(attempt)
                except ValueError:  # Non-integer code
                    log_method = self.logger.warning
                    is_retryable = True  # Default retry if code parsing fails
                    message = f"Bybit Exchange Error (Non-integer code: {exchange_specific_code}) for {item_desc}"
            # General checks for other exchanges or if Bybit code is not specific enough
            elif any(
                p in err_lower
                for p in [
                    "symbol invalid",
                    "market unavailable",
                    "bad symbol",
                    "parameter error",
                    "insufficient",
                    "balance",
                    "margin error",
                    "permission denied",
                ]
            ):
                is_retryable = False
            else:  # Default to retry for other exchange errors
                log_method = self.logger.warning
                is_retryable = True
                delay_seconds = _exponential_backoff(attempt)
        else:  # Unhandled or unexpected exceptions
            message = f"Unexpected error while fetching {item_desc}"
            is_retryable = False  # Do not retry unknown errors by default

        # Log the error with appropriate level and color
        log_method(
            f"{NEON_YELLOW if is_retryable else NEON_RED}"
            f"{message}: {exception_details} (Code: {exchange_specific_code or 'N/A'}) "
            f"(Attempt {attempt + 1}/{total_attempts}) Context: {context_info or ''}"
            f"{RESET_ALL_STYLE}",
            exc_info=(
                not is_retryable or log_method == self.logger.error
            ),  # Show stack trace for non-retryable or critical retryable errors
        )

        is_last_attempt = attempt == total_attempts - 1
        if not is_retryable or is_last_attempt:
            if is_retryable and is_last_attempt:  # Log final failure after retries
                self.logger.error(f"Final attempt failed for {item_desc} after {total_attempts} tries.")

            # Increment circuit breaker failure count if the operation ultimately failed
            self.circuit_breaker_failure_count += 1
            if (
                not self.circuit_breaker_tripped
                and self.circuit_breaker_failure_count >= self.circuit_breaker_max_failures
            ):
                self.circuit_breaker_tripped = True
                self.circuit_breaker_reset_time = time.monotonic() + self.circuit_breaker_cooldown
                self.logger.critical(
                    f"{NEON_RED}Circuit breaker TRIPPED for {self.circuit_breaker_cooldown:.0f} seconds due to repeated API failures.{RESET_ALL_STYLE}"
                )
            return False  # Do not retry

        # If retryable and not the last attempt
        if is_retryable:
            jitter = random.uniform(0, 0.2 * delay_seconds)  # Add up to 20% jitter
            actual_wait_time = delay_seconds + jitter
            self.logger.debug(f"Waiting {actual_wait_time:.2f}s before retrying {item_desc} (Attempt {attempt + 2})...")
            await asyncio.sleep(actual_wait_time)
            return True  # Retry

        return False  # Should not be reached if logic is correct

    # ==========================================================================
    # Market Data Methods
    # ==========================================================================
    async def load_markets(self, reload: bool = False) -> bool:
        """
        Loads or reloads market information from the exchange, updating the internal cache.

        Args:
            reload: If True, forces a reload even if cache is fresh.

        Returns:
            True if markets were successfully loaded or cache is valid, False otherwise.
        """
        current_time = time.monotonic()
        cache_is_valid = (
            self.markets_cache and (current_time - self.last_markets_update_time) < self.market_cache_duration_seconds
        )

        if not reload and cache_is_valid:
            self.logger.debug("Market data cache is fresh. Skipping reload.")
            return True

        params = self.exchange.safe_value(self.exchange.options, "loadMarketsParams", {})
        total_attempts = self.max_api_retries + 1  # Includes initial attempt

        for attempt in range(total_attempts):
            try:
                self.logger.info(f"Loading markets (Attempt {attempt + 1}/{total_attempts})...")
                # The `reload=True` for `exchange.load_markets` ensures CCXT fetches fresh data
                loaded_markets = await self.exchange.load_markets(reload=True, params=params)

                if not loaded_markets:
                    # This case might indicate an issue with the exchange response or CCXT parsing
                    raise ccxt_async.ExchangeError("load_markets returned an empty or invalid market structure.")

                self.markets_cache = loaded_markets
                self.last_markets_update_time = current_time
                self.logger.info(
                    f"{NEON_GREEN}Successfully loaded {len(self.markets_cache)} markets for {self.exchange.id}.{RESET_ALL_STYLE}"
                )

                # Optionally, process all markets immediately after loading
                # for symbol_key, market_data_val in self.markets_cache.items(): # Use different var names
                #    self._process_and_cache_market(symbol_key, market_data_val)
                # self.logger.info("Initial processing of all loaded markets complete.")

                return True
            except Exception as e:
                # Use the centralized exception handler
                should_retry = await self._handle_fetch_exception(
                    e, attempt, total_attempts, "loading markets", self.exchange.id
                )
                if not should_retry:
                    self.logger.critical(
                        f"{NEON_RED}Permanently failed to load markets for {self.exchange.id} after {attempt + 1} attempt(s).{RESET_ALL_STYLE}"
                    )
                    return False

        # If loop finishes, all retries failed
        self.logger.critical(
            f"{NEON_RED}Failed to load markets for {self.exchange.id} after all {total_attempts} retries.{RESET_ALL_STYLE}"
        )
        return False

    def _process_and_cache_market(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Internal helper to process raw market data, add derived fields, and update the cache.
        This ensures consistency in market object structure used by other methods.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').
            market_data: The raw market data dictionary from CCXT.

        Returns:
            The processed market data dictionary, or None if processing fails.
        """
        try:
            # Basic validation for essential fields
            required_fields = ["id", "symbol", "precision", "limits"]
            if not market_data or not all(field in market_data for field in required_fields):
                missing = [
                    field
                    for field in required_fields
                    if not market_data or field not in market_data or market_data.get(field) is None
                ]
                self.logger.error(f"Market data for {symbol} is missing essential fields: {missing}. Cannot process.")
                return None

            # Ensure defaults for precision and limits to prevent downstream errors
            market_data.setdefault("precision", {"price": "1e-8", "amount": "1e-8"})  # Default if 'precision' is None
            market_data["precision"].setdefault("price", "1e-8")  # Default if 'price' precision is None
            market_data["precision"].setdefault("amount", "1e-8")  # Default if 'amount' precision is None

            market_data.setdefault(
                "limits", {"amount": {"min": "0"}, "cost": {"min": "0"}, "price": {"min": "0", "max": None}}
            )
            market_data["limits"].setdefault("amount", {}).setdefault("min", "0")
            market_data["limits"].setdefault("cost", {}).setdefault("min", "0")
            market_data["limits"].setdefault("price", {}).setdefault("min", "0")

            # Determine if it's a contract market (futures, swaps)
            market_type_str = str(market_data.get("type", "unknown")).lower()
            is_contract = (
                market_data.get("contract", False)
                or market_type_str in ["swap", "future", "option", "futures"]
                or market_data.get("linear", False)
                or market_data.get("inverse", False)
            )
            market_data["is_contract"] = is_contract
            market_data["is_linear_contract"] = market_data.get("linear", False) and is_contract
            market_data["is_inverse_contract"] = market_data.get("inverse", False) and is_contract

            # Calculate and cache precision places and min tick size
            # These rely on utility functions that should handle potential errors
            market_data["pricePrecisionPlaces"] = get_price_precision(market_data, self.logger)
            market_data["minTickSizeDecimal"] = get_min_tick_size(market_data, self.logger)

            # Derive amount precision places if not directly available
            if "amountPrecisionPlaces" not in market_data:
                amount_precision_step = market_data["precision"].get("amount")
                derived_amount_precision_places = 8  # Default fallback
                if amount_precision_step:
                    try:
                        # Calculate places from the step size (e.g., '0.001' -> 3)
                        derived_amount_precision_places = abs(
                            Decimal(str(amount_precision_step)).normalize().as_tuple().exponent
                        )
                    except (InvalidOperation, TypeError, ValueError):
                        self.logger.warning(
                            f"Could not derive amount precision for {symbol} from step '{amount_precision_step}'. Defaulting to 8."
                        )
                market_data["amountPrecisionPlaces"] = derived_amount_precision_places

            # Update the main cache with the processed market data
            self.markets_cache[symbol] = market_data
            return market_data

        except Exception as e:
            self.logger.error(f"Error processing market data for {symbol}: {e}", exc_info=True)
            # Optionally, remove or mark this market as problematic in the cache
            if symbol in self.markets_cache:
                del self.markets_cache[symbol]
            return None

    async def get_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves processed market information for a given symbol.
        Uses cached data if available and fresh, otherwise loads/reloads markets
        and processes the specific symbol's data.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').

        Returns:
            A dictionary containing processed market information, or None if not found or error.
        """
        current_time = time.monotonic()
        cache_is_valid = (current_time - self.last_markets_update_time) < self.market_cache_duration_seconds

        if self.markets_cache and symbol in self.markets_cache and cache_is_valid:
            market = self.markets_cache[symbol]
            # Ensure essential processed fields are present; re-process if not
            if "pricePrecisionPlaces" in market and "minTickSizeDecimal" in market:
                return market
            else:
                self.logger.warning(f"Market {symbol} found in cache but lacks processed fields. Re-processing.")
                return self._process_and_cache_market(symbol, market)  # Re-process from cache

        # Cache miss or stale: attempt to load/reload markets
        if not await self.load_markets(reload=not cache_is_valid):  # Reload if stale or forced
            self.logger.error(f"Failed to load markets. Cannot retrieve info for {symbol}.")
            return None

        # After load_markets, try to get from updated cache
        market_raw = self.markets_cache.get(symbol)

        if not market_raw:  # If still not in cache, try direct CCXT market method as a fallback
            if hasattr(self.exchange, "market") and callable(self.exchange.market):
                try:
                    market_raw = self.exchange.market(symbol)
                except ccxt_async.BadSymbol:
                    self.logger.error(f"Market {symbol} not found via exchange.market() method (BadSymbol).")
                    return None
                except Exception as e:
                    self.logger.error(f"Error fetching market {symbol} via exchange.market(): {e}")
                    return None
            if not market_raw:  # If still not found
                self.logger.error(
                    f"Market {symbol} not found in cache or via exchange.market() after (re)load attempt."
                )
                return None

        # Process the raw market data
        processed_market = self._process_and_cache_market(market_raw.get("symbol", symbol), market_raw)
        if processed_market:
            # Ensure it's stored under the correct canonical symbol key in the cache
            self.markets_cache[processed_market["symbol"]] = processed_market
        return processed_market

    async def fetch_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Fetches the current price for a symbol using the ticker.
        Validates symbol and uses retry logic. Prioritizes 'last', then mid-price, then 'ask' or 'bid'.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').

        Returns:
            The current price as a Decimal, or None if fetching fails or no valid price found.
        """
        market_info = await self.get_market_info(symbol)
        if not market_info:
            self.logger.error(f"Cannot fetch price for {symbol}: market info not available.")
            return None

        total_attempts = self.max_api_retries + 1
        for attempt in range(total_attempts):
            try:
                self.logger.debug(f"Fetching ticker for {symbol} (Attempt {attempt + 1}/{total_attempts})")

                params = {}
                # For Bybit, specify category (linear/spot) for V5 API
                if self.exchange.id == "bybit":
                    # Determine category based on processed market_info
                    if market_info.get("is_linear_contract") or market_info.get(
                        "is_inverse_contract"
                    ):  # Covers both linear and inverse
                        params["category"] = "linear" if market_info.get("is_linear_contract") else "inverse"
                    elif market_info.get("spot", False) or market_info.get("type") == "spot":
                        params["category"] = "spot"
                    # else: rely on defaultType or CCXT's inference

                ticker = await self.exchange.fetch_ticker(symbol, params=params)
                if not ticker:
                    raise ccxt_async.ExchangeError(f"fetch_ticker for {symbol} returned an empty response.")

                # Try to find a valid price from the ticker data, in order of preference
                price_candidates_str = [
                    ticker.get("last"),  # Last traded price
                    ticker.get("close"),  # Same as last for many exchanges
                ]

                bid_str, ask_str = ticker.get("bid"), ticker.get("ask")
                if bid_str is not None and ask_str is not None:
                    try:
                        bid_dec, ask_dec = Decimal(str(bid_str)), Decimal(str(ask_str))
                        if bid_dec > 0 and ask_dec > 0:
                            price_candidates_str.append(str((bid_dec + ask_dec) / Decimal("2")))  # Mid-price
                    except (InvalidOperation, TypeError, ValueError):
                        self.logger.debug(
                            f"Could not calculate mid-price for {symbol} from bid/ask: {bid_str}/{ask_str}"
                        )

                price_candidates_str.extend([ask_str, bid_str])  # Fallback to ask or bid

                for price_str_val in price_candidates_str:  # Use different var name
                    if price_str_val is not None:
                        try:
                            price_decimal = Decimal(str(price_str_val))
                            if price_decimal > Decimal("0"):
                                self.logger.debug(f"Current price for {symbol}: {price_decimal}")
                                return price_decimal
                        except (InvalidOperation, TypeError, ValueError):
                            continue  # Try next candidate

                raise ccxt_async.ExchangeError(
                    f"No valid price (last, close, mid, ask, bid) found in ticker for {symbol}."
                )

            except Exception as e:
                should_retry = await self._handle_fetch_exception(
                    e, attempt, total_attempts, f"current price for {symbol}", symbol
                )
                if not should_retry:
                    self.logger.error(f"Permanently failed to fetch current price for {symbol}.")
                    return None

        self.logger.error(f"Failed to fetch current price for {symbol} after all retries.")
        return None

    async def fetch_current_prices(self, symbols: List[str]) -> Dict[str, Optional[Decimal]]:
        """
        Fetches current prices for multiple symbols concurrently.

        Args:
            symbols: A list of trading symbols.

        Returns:
            A dictionary mapping symbols to their current prices (Decimal) or None if failed.
        """
        if not symbols:
            return {}

        tasks = [self.fetch_current_price(s) for s in symbols]
        # return_exceptions=True allows us to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices: Dict[str, Optional[Decimal]] = {}
        for i, symbol_item in enumerate(symbols):  # Use different var name
            result = results[i]
            if isinstance(result, Decimal):
                prices[symbol_item] = result
            elif isinstance(result, Exception):
                self.logger.error(f"Failed to fetch price for {symbol_item} in batch operation: {result}")
                prices[symbol_item] = None
            else:  # Should be None if fetch_current_price returned None without exception
                prices[symbol_item] = None
        return prices

    async def fetch_klines(
        self, symbol: str, timeframe: str, limit: int = 250, since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetches OHLCV kline data for a symbol and timeframe.
        Validates symbol, timeframe, and handles response parsing into a pandas DataFrame.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').
            timeframe: The kline timeframe (e.g., '1m', '1h', '1d').
            limit: The number of klines to fetch.
            since: Timestamp (ms) to fetch klines since.

        Returns:
            A pandas DataFrame with OHLCV data, indexed by timestamp, or an empty DataFrame on failure.
            Columns: 'open', 'high', 'low', 'close', 'volume' (all as Decimal).
        """
        market_info = await self.get_market_info(symbol)
        if not market_info:
            self.logger.error(f"Cannot fetch klines for {symbol}: market info not available.")
            return pd.DataFrame()

        if not self.exchange.has.get("fetchOHLCV"):
            self.logger.error(f"Exchange {self.exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        if timeframe not in self.exchange.timeframes:
            self.logger.error(
                f"Invalid timeframe '{timeframe}' for {self.exchange.id}. Available: {list(self.exchange.timeframes.keys()) if self.exchange.timeframes else 'None'}"
            )
            return pd.DataFrame()

        total_attempts = self.max_api_retries + 1
        for attempt in range(total_attempts):
            try:
                params = {}
                # For Bybit, specify category for V5 API
                if self.exchange.id == "bybit":
                    if market_info.get("is_linear_contract") or market_info.get("is_inverse_contract"):
                        params["category"] = "linear" if market_info.get("is_linear_contract") else "inverse"
                    elif market_info.get("spot", False) or market_info.get("type") == "spot":
                        params["category"] = "spot"

                self.logger.debug(
                    f"Fetching klines for {symbol} [{timeframe}], Limit: {limit}, Since: {since} (Attempt {attempt + 1})"
                )
                # CCXT fetch_ohlcv returns a list of lists: [timestamp, open, high, low, close, volume]
                ohlcv_data = await self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=limit, since=since, params=params
                )

                if (
                    ohlcv_data
                    and isinstance(ohlcv_data, list)
                    and all(isinstance(row, list) and len(row) >= 6 for row in ohlcv_data)
                ):
                    df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

                    # Convert timestamp to datetime, ensuring UTC and then removing timezone for naive datetime
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="ms", errors="coerce", utc=True
                    ).dt.tz_localize(None)
                    df.dropna(subset=["timestamp"], inplace=True)  # Remove rows where timestamp conversion failed
                    df.set_index("timestamp", inplace=True)

                    # Convert OHLCV columns to Decimal for precision, handling potential non-numeric values
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)

                    # Basic data cleaning: drop rows with NA in key price fields or negative values
                    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                    cleaned_df = df[(df["close"] > Decimal(0)) & (df["volume"] >= Decimal(0))].copy()

                    if cleaned_df.empty:
                        if len(df) > 0:  # Data was fetched but all filtered out
                            self.logger.warning(
                                f"All {len(df)} klines fetched for {symbol} [{timeframe}] were filtered out by cleaning criteria."
                            )
                        else:  # No data returned from exchange that passed initial checks
                            self.logger.info(
                                f"No valid kline data returned for {symbol} [{timeframe}] that passed initial parsing."
                            )
                        return pd.DataFrame()  # Return empty if all are filtered

                    cleaned_df.sort_index(inplace=True)  # Ensure chronological order
                    self.logger.info(
                        f"Successfully fetched and processed {len(cleaned_df)} klines for {symbol} [{timeframe}]."
                    )
                    return cleaned_df

                elif isinstance(ohlcv_data, list) and len(ohlcv_data) == 0 and limit > 0:  # No data returned
                    self.logger.info(
                        f"No kline data available for {symbol} [{timeframe}] for the requested period/limit."
                    )
                    return pd.DataFrame()
                else:
                    raise ccxt_async.ExchangeError(
                        f"fetch_ohlcv for {symbol} [{timeframe}] returned invalid or unexpected data format: {type(ohlcv_data)}"
                    )

            except Exception as e:
                item_description = f"klines for {symbol} [{timeframe}]"
                should_retry = await self._handle_fetch_exception(e, attempt, total_attempts, item_description, symbol)
                if not should_retry:
                    self.logger.error(f"Permanently failed to fetch {item_description}.")
                    return pd.DataFrame()

        self.logger.error(f"Failed to fetch klines for {symbol} [{timeframe}] after all retries.")
        return pd.DataFrame()

    async def fetch_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict[str, Any]]:
        """
        Fetches the order book for a symbol.
        Validates symbol and handles empty or malformed order book responses.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').
            limit: The maximum number of bids/asks to fetch.

        Returns:
            A dictionary representing the order book (CCXT format), or None on failure.
            Includes 'bids': [[price, amount]], 'asks': [[price, amount]], 'timestamp', 'datetime', 'nonce', 'symbol'.
        """
        market_info = await self.get_market_info(symbol)
        if not market_info:
            self.logger.error(f"Cannot fetch order book for {symbol}: market info not available.")
            return None

        if not self.exchange.has.get("fetchOrderBook"):
            self.logger.error(f"Exchange {self.exchange.id} does not support fetchOrderBook.")
            return None

        total_attempts = self.max_api_retries + 1
        for attempt in range(total_attempts):
            try:
                params = {}
                # For Bybit, specify category for V5 API
                if self.exchange.id == "bybit":
                    if market_info.get("is_linear_contract") or market_info.get("is_inverse_contract"):
                        params["category"] = "linear" if market_info.get("is_linear_contract") else "inverse"
                    elif market_info.get("spot", False) or market_info.get("type") == "spot":
                        params["category"] = "spot"

                self.logger.debug(f"Fetching order book for {symbol}, Limit: {limit} (Attempt {attempt + 1})")
                order_book = await self.exchange.fetch_order_book(symbol, limit=limit, params=params)

                # Validate structure (basic check for bids, asks as lists)
                if (
                    order_book
                    and isinstance(order_book, dict)
                    and "bids" in order_book
                    and isinstance(order_book["bids"], list)
                    and "asks" in order_book
                    and isinstance(order_book["asks"], list)
                ):
                    # Handle cases where exchange might return an empty but valid structure
                    if not order_book["bids"] and not order_book["asks"]:
                        self.logger.info(f"Order book for {symbol} is currently empty (no bids or asks).")

                    # Ensure standard CCXT fields are present
                    current_ts = self.exchange.milliseconds()
                    order_book.setdefault("timestamp", current_ts)
                    order_book.setdefault(
                        "datetime", self.exchange.iso8601(order_book["timestamp"])
                    )  # Use OB timestamp if present
                    order_book.setdefault("nonce", None)  # Nonce might not be relevant for all exchanges/orderbooks
                    order_book.setdefault("symbol", symbol)  # Ensure symbol is in the returned dict

                    # Optional: Convert prices/amounts to Decimal if not already done by CCXT
                    # for side_key in ['bids', 'asks']: # Use different var name
                    #     order_book[side_key] = [[Decimal(str(p)), Decimal(str(a))] for p, a in order_book[side_key]]

                    self.logger.debug(f"Successfully fetched order book for {symbol}.")
                    return order_book
                else:
                    raise ccxt_async.ExchangeError(
                        f"fetch_order_book for {symbol} returned an invalid or malformed structure: {type(order_book)}"
                    )

            except Exception as e:
                item_description = f"order book for {symbol}"
                should_retry = await self._handle_fetch_exception(e, attempt, total_attempts, item_description, symbol)
                if not should_retry:
                    self.logger.error(f"Permanently failed to fetch {item_description}.")
                    return None

        self.logger.error(f"Failed to fetch order book for {symbol} after all retries.")
        return None

    # --- Account Data Methods ---
    async def fetch_balance(self, currency: Optional[str] = None) -> Union[Optional[Decimal], Optional[Dict[str, Any]]]:
        """
        Fetches account balance. If currency is specified, returns its 'free' balance as Decimal.
        Otherwise, returns the full balance structure from CCXT.
        Handles Bybit V5 specific parameters and response parsing.

        Args:
            currency: Optional. The specific currency code (e.g., 'USDT') to get the balance for.

        Returns:
            - If currency is specified: Free balance of that currency as Decimal, or None on failure/not found.
            - If currency is None: Full balance dictionary (CCXT format), or None on failure.
        """
        # Prepare parameters, especially for Bybit V5
        request_params = self.exchange.safe_value(self.exchange.options, "balanceFetchParams", {}).copy()
        context_description = f"balance for {currency.upper()}" if currency else "all account balances"

        if self.exchange.id == "bybit":
            # Determine accountType if not explicitly set in params
            if "accountType" not in request_params:
                # Use default_market_type from config to infer accountType
                default_mkt_type = self._config.get("default_market_type", "unified").lower()
                if default_mkt_type == "unified":
                    request_params["accountType"] = "UNIFIED"
                elif default_mkt_type in ["swap", "future", "futures", "linear", "inverse"]:  # Contract types
                    request_params["accountType"] = "CONTRACT"
                elif default_mkt_type == "spot":
                    request_params["accountType"] = "SPOT"
                # else: rely on CCXT default or exchange default if not specified

            # If fetching for a specific coin and accountType is UNIFIED or CONTRACT, Bybit V5 needs 'coin' param
            if currency and "coin" not in request_params:
                acc_type = request_params.get("accountType")
                if acc_type in ["UNIFIED", "CONTRACT"]:  # Bybit V5 specific
                    request_params["coin"] = currency.upper()

        currency_upper = currency.upper() if currency else None
        total_attempts = self.max_api_retries + 1

        for attempt in range(total_attempts):
            try:
                self.logger.debug(f"Fetching {context_description} (Attempt {attempt + 1}). Params: {request_params}")
                balance_info = await self.exchange.fetch_balance(params=request_params)

                if not balance_info:  # Should not happen if call is successful, CCXT returns at least {}
                    raise ccxt_async.ExchangeError("fetch_balance returned an empty or null response.")

                # If no specific currency, return the whole balance structure
                if not currency_upper:
                    self.logger.info(f"Successfully fetched full balance data. Keys: {list(balance_info.keys())}")
                    return balance_info

                # Attempt to get free balance for the specified currency
                # Standard CCXT structure: balance_info[currency_upper]['free']
                free_balance_str = self.exchange.safe_string(balance_info.get(currency_upper, {}), "free")

                # Bybit V5 specific parsing if standard path fails (especially for UNIFIED/CONTRACT with 'coin' param)
                if free_balance_str is None and self.exchange.id == "bybit":
                    # Bybit V5 often returns detailed list under 'info.result.list'
                    bybit_info_list = self.exchange.safe_value(balance_info, ["info", "result", "list"], [])
                    if bybit_info_list and isinstance(bybit_info_list, list) and len(bybit_info_list) > 0:
                        # Expected accountType for Bybit
                        target_account_type = request_params.get("accountType")

                        for account_detail in bybit_info_list:
                            if not isinstance(account_detail, dict):
                                continue

                            # Match accountType if specified
                            current_acc_type = self.exchange.safe_string(account_detail, "accountType")
                            if target_account_type and current_acc_type != target_account_type:
                                continue

                            # Find the coin in this account's details
                            coin_list = self.exchange.safe_value(account_detail, "coin", [])
                            for coin_detail in coin_list:
                                if not isinstance(coin_detail, dict):
                                    continue
                                if self.exchange.safe_string(coin_detail, "coin") == currency_upper:
                                    # Prefer 'availableToWithdraw' or 'availableBalance' for UNIFIED/CONTRACT
                                    if target_account_type in ["UNIFIED", "CONTRACT"]:
                                        free_balance_str = self.exchange.safe_string_2(
                                            coin_detail, "availableToWithdraw", "availableBalance"
                                        )
                                    # For SPOT, it might be 'availableBal' or similar
                                    elif target_account_type == "SPOT":
                                        free_balance_str = self.exchange.safe_string(
                                            coin_detail, "availableBal"
                                        )  # Check Bybit docs for exact field

                                    if free_balance_str is not None:
                                        break  # Found
                            if free_balance_str is not None:
                                break  # Found

                        # Fallback for SPOT if not found via specific accountType matching (e.g. if accountType wasn't in params)
                        # This assumes the first entry in 'list' might be the relevant one for SPOT if no explicit type given
                        if free_balance_str is None and (not target_account_type or target_account_type == "SPOT"):
                            if isinstance(bybit_info_list[0], dict):
                                coin_list_spot = self.exchange.safe_value(bybit_info_list[0], "coin", [])
                                for coin_detail_spot in coin_list_spot:
                                    if self.exchange.safe_string(coin_detail_spot, "coin") == currency_upper:
                                        free_balance_str = self.exchange.safe_string(
                                            coin_detail_spot, "availableBal"
                                        )  # Path for SPOT
                                        if free_balance_str is not None:
                                            break

                # Fallback for general CCXT structure if still not found (e.g. balance_info['free'][currency_upper])
                if free_balance_str is None:
                    free_balance_str = self.exchange.safe_string(balance_info.get("free", {}), currency_upper)

                if free_balance_str is not None:
                    try:
                        balance_decimal = Decimal(free_balance_str)
                        self.logger.info(f"Successfully fetched free balance for {currency_upper}: {balance_decimal}")
                        return balance_decimal
                    except InvalidOperation:
                        raise ccxt_async.ExchangeError(
                            f"Could not parse free balance string '{free_balance_str}' for {currency_upper} into Decimal."
                        )
                else:
                    # If currency was specified but not found after all checks
                    self.logger.warning(
                        f"Currency '{currency_upper}' not found in balance response. Available currencies might be: {list(balance_info.keys())}"
                    )
                    # Consider if this should raise an error or return None. Returning None indicates "not found".
                    # If an error occurred that prevented finding it, _handle_fetch_exception would catch it.
                    # This path means the API call succeeded, but the currency isn't in the response as expected.
                    return None  # Currency not found in the balance structure

            except Exception as e:
                should_retry = await self._handle_fetch_exception(
                    e, attempt, total_attempts, context_description, currency_upper or self.exchange.id
                )
                if not should_retry:
                    self.logger.error(f"Permanently failed to fetch {context_description}.")
                    return None

        self.logger.error(f"Failed to fetch {context_description} after all retries.")
        return None

    async def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches and processes the open position for a given contract symbol.
        Standardizes key fields like size, side, entry price, etc., into Decimal types.

        Args:
            symbol: The trading symbol for the contract (e.g., 'BTC/USDT:USDT').

        Returns:
            A dictionary representing the processed open position, or None if no active position
            or an error occurs. Includes standardized Decimal fields like 'contractsDecimal',
            'entryPriceDecimal', etc.
        """
        if not self.exchange.has.get("fetchPositions"):
            self.logger.warning(
                f"Exchange {self.exchange.id} does not support fetchPositions. Cannot get open position for {symbol}."
            )
            return None

        market_info = await self.get_market_info(symbol)
        if not market_info:
            self.logger.error(f"Cannot get position for {symbol}: market info not available.")
            return None
        if not market_info.get("is_contract"):
            self.logger.info(f"{symbol} is not a contract. No position to fetch.")
            return None

        # Prepare parameters for fetch_positions, especially for Bybit V5
        fetch_pos_params = self.exchange.safe_value(self.exchange.options, "fetchPositionsParams", {}).copy()
        if self.exchange.id == "bybit":
            # Determine category: linear or inverse
            if "category" not in fetch_pos_params:
                if market_info.get("is_linear_contract"):
                    fetch_pos_params["category"] = "linear"
                elif market_info.get("is_inverse_contract"):
                    fetch_pos_params["category"] = "inverse"
                else:
                    self.logger.error(
                        f"Cannot determine Bybit contract category (linear/inverse) for {symbol}. Market type: {market_info.get('type')}"
                    )
                    return None

            # Bybit V5 fetchPositions can filter by symbol if category is provided
            fetch_pos_params["symbol"] = market_info.get("id", symbol)  # Use market ID

        positions_data: List[Dict[str, Any]] = []
        total_attempts = self.max_api_retries + 1
        for attempt in range(total_attempts):
            try:
                self.logger.debug(f"Fetching position for {symbol} (Attempt {attempt + 1}). Params: {fetch_pos_params}")

                # CCXT fetch_positions can take a list of symbols or None (for all)
                # If Bybit and symbol is in params, it implies fetching for that specific symbol.
                # Some exchanges might require `symbols=[symbol]` even if params also specify symbol.
                if self.exchange.id == "bybit" and "symbol" in fetch_pos_params:
                    # Bybit V5 with category and symbol in params should return only that position.
                    # Passing symbols=None is typical here.
                    positions_data = await self.exchange.fetch_positions(symbols=None, params=fetch_pos_params)
                elif self.exchange.has.get("fetchPositions") is True:  # Explicit check for method support
                    # Standard way if exchange supports symbol list
                    positions_data = await self.exchange.fetch_positions(symbols=[symbol], params=fetch_pos_params)
                else:  # Fallback if specific symbol fetching isn't directly supported, fetch all and filter
                    all_positions = await self.exchange.fetch_positions(params=fetch_pos_params)
                    positions_data = [p for p in all_positions if self.exchange.safe_string(p, "symbol") == symbol]

                break  # Success
            except Exception as e:
                # Special handling for "position does not exist" (Bybit code 110025)
                if self.exchange.id == "bybit":
                    bybit_err_code = self.exchange.safe_string(getattr(e, "info", {}), "retCode")
                    if bybit_err_code == "110025":  # "position is not exists"
                        self.logger.info(f"No open position found for {symbol} via API (Code 110025).")
                        return None  # This is a valid state, not an error to retry.

                should_retry = await self._handle_fetch_exception(
                    e, attempt, total_attempts, f"position data for {symbol}", symbol
                )
                if not should_retry:
                    self.logger.error(f"Permanently failed to fetch position for {symbol}.")
                    return None
        else:  # Loop completed without break (all retries failed)
            self.logger.error(f"Failed to fetch position for {symbol} after all retries.")
            return None

        if not positions_data:
            self.logger.info(f"No position structures returned for {symbol} from API.")
            return None

        active_position: Optional[Dict[str, Any]] = None
        # Define a small threshold for position size to filter out "dust" or effectively closed positions
        amount_precision_places = market_info.get("amountPrecisionPlaces", 8)  # Fallback to 8
        size_threshold = Decimal(f"1e-{amount_precision_places + 1}")  # One order of magnitude smaller than precision

        for pos_item in positions_data:
            if not isinstance(pos_item, dict):
                self.logger.warning(f"Skipping non-dictionary item in positions_data for {symbol}: {pos_item}")
                continue

            # Get position size using multiple common keys from CCXT structure or info field
            # 'contracts' is the CCXT standard field for contract size (absolute value)
            # 'info.size' or 'info.qty' are common in Bybit raw responses
            size_str = self.exchange.safe_string_n(
                pos_item,
                [
                    "contracts",  # CCXT standard (absolute value)
                    ("info", "size"),  # Bybit V5
                    ("info", "contracts"),  # Alternative CCXT field some exchanges might populate
                    ("info", "qty"),  # Common in some exchanges raw data
                ],
            )

            if size_str is None:
                self.logger.debug(f"Could not determine size for a position item of {symbol}. Item: {pos_item}")
                continue

            try:
                position_size_abs = Decimal(size_str)  # This should be absolute size

                # For Bybit V5, 'side' might be 'None' if position is flat.
                # 'positionSide' ('Buy'/'Sell') or 'side' ('Buy'/'Sell') in 'info' can indicate actual side.
                # A non-zero 'size' in 'info' usually means an open position.
                # If CCXT's 'contracts' is zero or very small, it's likely flat.
                if self.exchange.id == "bybit":
                    bybit_v5_info_side = self.exchange.safe_string_lower(
                        pos_item, ("info", "side")
                    )  # 'Buy', 'Sell', or 'None'
                    # bybit_v5_info_position_idx = self.exchange.safe_integer(pos_item, ('info', 'positionIdx')) # 0 for one-way, 1 for Buy hedge, 2 for Sell hedge

                    # If Bybit info.size implies zero and CCXT contracts is also zero/small, skip
                    if (
                        abs(position_size_abs) <= size_threshold
                        and self.exchange.safe_string_lower(pos_item, ("info", "size")) == "0"
                    ):
                        if bybit_v5_info_side == "none":  # Bybit V5 can have side 'None' for flat one-way positions
                            self.logger.debug(
                                f"Skipping effectively flat Bybit position for {symbol} (size: {size_str}, info.side: 'None')."
                            )
                            continue

                if abs(position_size_abs) <= size_threshold:  # Filter out effectively zero positions
                    self.logger.debug(
                        f"Skipping position item for {symbol} with size {position_size_abs} (<= threshold {size_threshold})."
                    )
                    continue

                # Process this item as a potentially active position
                processed_pos = pos_item.copy()  # Work on a copy
                processed_pos["contractsDecimal"] = position_size_abs  # Standardized absolute size

                # Determine position side ('long' or 'short')
                # CCXT 'side' field should be 'long' or 'short'. If not, infer.
                current_side = self.exchange.safe_string_lower(processed_pos, "side")

                if not current_side or current_side == "none":  # 'none' can appear in Bybit V5
                    if self.exchange.id == "bybit":
                        # Bybit V5: 'info.side' ('Buy'/'Sell') or 'info.positionSide' ('Buy'/'Sell')
                        # Note: 'info.side' refers to the side of the position (Buy for long, Sell for short).
                        # 'info.positionSide' is for hedge mode (Buy for long leg, Sell for short leg).
                        # For one-way mode, 'info.positionSide' might be 'Both'.
                        # We rely on 'info.side' primarily for direction if CCXT 'side' is ambiguous.

                        info_side_bybit = self.exchange.safe_string_lower(
                            processed_pos, ("info", "side")
                        )  # 'Buy' or 'Sell'
                        if info_side_bybit == "buy":
                            current_side = "long"
                        elif info_side_bybit == "sell":
                            current_side = "short"
                        else:
                            self.logger.warning(
                                f"Could not reliably determine side for Bybit position on {symbol}. CCXT side: '{current_side}', info.side: '{info_side_bybit}'. Size: {position_size_abs}. Item: {pos_item}"
                            )
                            continue
                    else:
                        self.logger.warning(
                            f"Position for {symbol} has size {position_size_abs} but side is '{current_side}'. Cannot determine direction."
                        )
                        continue

                processed_pos["side"] = current_side  # Ensure 'side' is 'long' or 'short'

                # Standardize various numeric fields to Decimal
                fields_to_decimalize = {
                    "entryPriceDecimal": ["entryPrice", ("info", "avgPrice"), ("info", "entryPrice")],
                    "markPriceDecimal": ["markPrice", ("info", "markPrice")],
                    "liquidationPriceDecimal": ["liquidationPrice", ("info", "liqPrice")],
                    "unrealizedPnlDecimal": ["unrealizedPnl", ("info", "unrealisedPnl"), ("info", "unrealizedPnl")],
                    "leverageDecimal": ["leverage", ("info", "leverage")],
                    "collateralDecimal": ["collateral", ("info", "positionMargin"), ("info", "collateral")],
                    "initialMarginDecimal": ["initialMargin", ("info", "imr"), ("info", "initialMargin")],
                    "maintenanceMarginDecimal": ["maintenanceMargin", ("info", "mmr"), ("info", "maintMargin")],
                    "positionValueDecimal": [("info", "positionValue")],
                    "stopLossPriceDecimal": ["stopLossPrice", ("info", "stopLoss")],
                    "takeProfitPriceDecimal": ["takeProfitPrice", ("info", "takeProfit")],
                    "trailingStopPriceDecimal": [("info", "trailingStop")],
                    "trailingStopActivationPriceDecimal": [("info", "activePrice")],
                }

                for target_key, source_paths in fields_to_decimalize.items():
                    value_str = self.exchange.safe_string_n(processed_pos, source_paths)  # Use processed_pos here
                    processed_pos[target_key] = None
                    if value_str is not None and value_str.strip() and value_str.lower() != "null":
                        is_protection_price = target_key in [
                            "stopLossPriceDecimal",
                            "takeProfitPriceDecimal",
                            "trailingStopPriceDecimal",
                            "trailingStopActivationPriceDecimal",
                        ]
                        try:
                            val_dec = Decimal(value_str)
                            if is_protection_price and val_dec == Decimal(0):
                                pass
                            else:
                                processed_pos[target_key] = val_dec
                        except InvalidOperation:
                            self.logger.warning(
                                f"Could not parse '{value_str}' into Decimal for {target_key} on {symbol}."
                            )

                # Timestamp and datetime
                ts_ms = self.exchange.safe_integer_product_n(
                    processed_pos,
                    [  # Use processed_pos
                        ("timestamp", 1),
                        ("info", "updatedTime", 1),
                        ("info", "updatedTimeStamp", 1),
                        ("info", "createdTime", 1),
                    ],
                )
                if ts_ms:
                    processed_pos["timestamp_ms"] = ts_ms
                    processed_pos["datetime"] = self.exchange.iso8601(ts_ms)

                if self.exchange.id == "bybit":
                    processed_pos["positionIdx"] = self.exchange.safe_integer(processed_pos, ("info", "positionIdx"))
                    processed_pos["marginMode"] = self.exchange.safe_string_lower(processed_pos, ("info", "tradeMode"))
                    if not processed_pos["marginMode"]:
                        processed_pos["marginMode"] = self.exchange.safe_string_lower(
                            processed_pos, ("info", "marginMode")
                        )

                active_position = processed_pos
                break

            except Exception as e_parse:
                self.logger.error(
                    f"Error parsing a position item for {symbol}: {e_parse}. Item: {pos_item}", exc_info=True
                )

        if active_position:
            self.logger.info(
                f"{NEON_GREEN}Active {active_position.get('side', 'UNKNOWN').upper()} position found for {symbol}: "
                f"Size={active_position.get('contractsDecimal')}, Entry={active_position.get('entryPriceDecimal')}{RESET_ALL_STYLE}"
            )
            return active_position
        else:
            self.logger.info(
                f"No active, significantly sized position found for {symbol} after checking {len(positions_data)} structure(s)."
            )
            return None

    # --- Trading Execution Methods ---
    async def set_leverage(self, symbol: str, leverage: Union[int, float, Decimal]) -> bool:
        """
        Sets leverage for a contract symbol.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT:USDT').
            leverage: The desired leverage value (e.g., 10 for 10x).

        Returns:
            True if leverage was set successfully or already at the desired value, False otherwise.
        """
        market_info = await self.get_market_info(symbol)
        if not market_info:
            self.logger.error(f"Cannot set leverage for {symbol}: market info not available.")
            return False
        if not market_info.get("is_contract"):
            self.logger.warning(f"Cannot set leverage for {symbol}: it is not a contract market.")
            return False

        try:
            leverage_float = float(leverage)
            if leverage_float <= 0:
                raise ValueError("Leverage must be positive.")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid leverage value '{leverage}': {e}. Must be a positive number.")
