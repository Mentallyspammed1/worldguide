# File: exchange_api.py
"""Module for interacting with cryptocurrency exchanges via CCXT with enhanced error handling, retries, and Bybit V5 API support."""

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Union
import asyncio  # For asyncio.sleep
import importlib.metadata  # For getting package version
import pandas as pd

# Use async support version of ccxt
import ccxt.async_support as ccxt_async  # Renamed to avoid conflict with standard ccxt if used elsewhere

# Import constants and utility functions
from utils import (
    MAX_API_RETRIES,
    NEON_GREEN,
    NEON_RED,
    NEON_YELLOW,
    RESET_ALL_STYLE as RESET,  # Use RESET_ALL_STYLE from utils
    RETRY_DELAY_SECONDS,
    get_min_tick_size,
    get_price_precision,
)

module_logger = logging.getLogger(__name__)
_market_info_cache: Dict[str, Dict[str, Any]] = {}


def _exponential_backoff(attempt: int, base_delay: float = RETRY_DELAY_SECONDS, max_cap: float = 60.0) -> float:
    """Calculate delay for exponential backoff with a cap."""
    delay = base_delay * (2**attempt)
    return min(delay, max_cap)


async def _handle_fetch_exception(
    e: Exception, logger: logging.Logger, attempt: int, total_attempts: int, item_desc: str, context_info: str
) -> bool:
    """Helper to log and determine if a fetch exception is retryable for async functions."""
    is_retryable = False
    current_retry_delay = RETRY_DELAY_SECONDS  # Default delay
    error_detail = str(e)

    if isinstance(e, (ccxt_async.NetworkError, ccxt_async.RequestTimeout, asyncio.TimeoutError)):
        log_level_method = logger.warning
        is_retryable = True
        msg = f"Network/Timeout error fetching {item_desc} for {context_info}"
    elif isinstance(e, (ccxt_async.RateLimitExceeded, ccxt_async.DDoSProtection)):
        log_level_method = logger.warning
        is_retryable = True
        msg = f"Rate limit/DDoS triggered fetching {item_desc} for {context_info}"
        # Use longer, exponential backoff for rate limits
        current_retry_delay = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS * 3, max_cap=180.0)
    elif isinstance(e, ccxt_async.ExchangeError):
        log_level_method = logger.error  # Default to error for ExchangeError
        is_retryable = False
        err_str_lower = error_detail.lower()
        # Phrases that usually indicate a non-retryable client-side or setup error for fetch operations
        non_retryable_phrases = [
            "symbol",
            "market",
            "not found",
            "invalid",
            "parameter",
            "argument",
            "orderid",
            "insufficient",
            "balance",
            "margin account not exist",
        ]
        # Bybit specific error codes that are often non-retryable for fetch operations
        non_retryable_codes = [
            10001,  # params error
            110025,  # position not found / not exist
            110009,  # margin account not exist
            110045,  # unified account not exist
        ]
        if (
            any(phrase in err_str_lower for phrase in non_retryable_phrases)
            or getattr(e, "code", None) in non_retryable_codes
        ):
            msg = f"Exchange error (likely non-retryable) fetching {item_desc} for {context_info}"
        else:
            # Some exchange errors might be temporary (e.g., temporary trading ban, internal server error)
            msg = f"Potentially temporary Exchange error fetching {item_desc} for {context_info}"
            log_level_method = logger.warning  # Downgrade to warning for retry
            is_retryable = True
    else:
        log_level_method = logger.error
        is_retryable = False
        msg = f"Unexpected error fetching {item_desc} for {context_info}"
        # Log with exc_info for unexpected errors to get traceback
        log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET}", exc_info=True)
        return False  # No retry for truly unexpected errors handled here

    log_level_method(
        f"{NEON_YELLOW if is_retryable else NEON_RED}{msg}: {error_detail} (Attempt {attempt + 1}/{total_attempts}){RESET}"
    )

    if is_retryable and attempt < total_attempts - 1:
        logger.warning(f"Waiting {current_retry_delay:.2f}s before retrying {item_desc} fetch for {context_info}...")
        await asyncio.sleep(current_retry_delay)
    return is_retryable


async def initialize_exchange(
    api_key: str, api_secret: str, config: Dict[str, Any], logger: logging.Logger
) -> Optional[ccxt_async.Exchange]:
    exchange: Optional[ccxt_async.Exchange] = None
    try:
        try:
            ccxt_version = importlib.metadata.version("ccxt")
            logger.info(f"Using CCXT version: {ccxt_version}")
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Could not determine CCXT version. Ensure 'ccxt' is installed.")

        exchange_id = config.get("exchange_id", "bybit").lower()
        if not hasattr(ccxt_async, exchange_id):
            logger.error(f"{NEON_RED}Exchange ID '{exchange_id}' not found in CCXT async library.{RESET}")
            return None

        exchange_class = getattr(ccxt_async, exchange_id)
        exchange_options = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,  # CCXT's built-in rate limiter
            "options": {
                "defaultType": config.get(
                    "default_market_type", "linear"
                ),  # e.g., 'linear', 'inverse', 'spot', 'unified'
                "adjustForTimeDifference": True,  # Auto-sync time with server
                # Timeouts for various operations (in milliseconds)
                "fetchTickerTimeout": config.get("ccxt_fetch_ticker_timeout_ms", 15000),
                "fetchBalanceTimeout": config.get("ccxt_fetch_balance_timeout_ms", 20000),
                "createOrderTimeout": config.get("ccxt_create_order_timeout_ms", 25000),
                "cancelOrderTimeout": config.get("ccxt_cancel_order_timeout_ms", 20000),
                "fetchPositionsTimeout": config.get("ccxt_fetch_positions_timeout_ms", 20000),
                "fetchOHLCVTimeout": config.get("ccxt_fetch_ohlcv_timeout_ms", 20000),
                "loadMarketsTimeout": config.get("ccxt_load_markets_timeout_ms", 30000),
            },
        }
        if exchange_id == "bybit":
            # Bybit: Market orders do not require a price parameter
            exchange_options["options"]["createOrderRequiresPrice"] = False

        exchange = exchange_class(exchange_options)

        if config.get("use_sandbox"):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            if hasattr(exchange, "set_sandbox_mode") and callable(exchange.set_sandbox_mode):
                try:
                    # Some exchanges have a method to switch to sandbox
                    await exchange.set_sandbox_mode(True)  # CCXT standard way
                    logger.info(f"Sandbox mode enabled for {exchange.id} via set_sandbox_mode(True).")
                except Exception as sandbox_err:
                    logger.warning(
                        f"Error calling set_sandbox_mode(True) for {exchange.id}: {sandbox_err}. "
                        f"Attempting manual URL override if known."
                    )
                    # Fallback for Bybit if set_sandbox_mode is problematic or not available
                    if exchange.id == "bybit":
                        testnet_url = exchange.urls.get("test", "https://api-testnet.bybit.com")
                        exchange.urls["api"] = testnet_url
                        logger.info(f"Manual Bybit testnet URL set: {testnet_url}")
            elif exchange.id == "bybit":  # Direct manual override for Bybit
                testnet_url = exchange.urls.get("test", "https://api-testnet.bybit.com")
                exchange.urls["api"] = testnet_url
                logger.info(f"Manual Bybit testnet URL override applied: {testnet_url}")
            else:
                logger.warning(
                    f"{NEON_YELLOW}{exchange.id} doesn't support set_sandbox_mode or known manual override. "
                    f"Ensure API keys are Testnet keys if using sandbox.{RESET}"
                )

        logger.info(f"Loading markets for {exchange.id}...")
        await exchange.load_markets(reload=True)  # reload=True ensures fresh market data
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox', False)}")

        quote_currency = config.get("quote_currency", "USDT")
        default_market_type = exchange.options.get("defaultType", "N/A")
        logger.info(f"Attempting initial balance fetch for {quote_currency} (context: {default_market_type})...")
        balance_decimal = await fetch_balance(exchange, quote_currency, logger)
        if balance_decimal is not None:
            logger.info(
                f"{NEON_GREEN}Initial balance fetch successful for {quote_currency}: {balance_decimal:.4f}{RESET}"
            )
        else:
            logger.error(
                f"{NEON_RED}Initial balance fetch FAILED for {quote_currency}. "
                f"Bot might not function correctly if balance is critical.{RESET}"
            )
        return exchange
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        if exchange:
            await exchange.close()
    return None


async def fetch_current_price_ccxt(
    exchange: ccxt_async.Exchange, symbol: str, logger: logging.Logger
) -> Optional[Decimal]:
    attempts = 0
    total_attempts = MAX_API_RETRIES + 1
    while attempts < total_attempts:
        try:
            logger.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1}/{total_attempts})")
            ticker = await exchange.fetch_ticker(symbol)

            price_sources = []
            if ticker.get("bid") is not None and ticker.get("ask") is not None:
                try:
                    bid = Decimal(str(ticker["bid"]))
                    ask = Decimal(str(ticker["ask"]))
                    if bid > 0 and ask > 0 and ask >= bid:
                        price_sources.append((bid + ask) / Decimal("2"))  # Mid-price
                except (InvalidOperation, TypeError):
                    logger.debug(f"Could not parse bid/ask for mid-price for {symbol}.", exc_info=True)

            for key in ["last", "close", "ask", "bid"]:
                if ticker.get(key) is not None:
                    price_sources.append(ticker[key])

            for price_val in price_sources:
                if price_val is not None:
                    try:
                        price_dec = Decimal(str(price_val))
                        if price_dec > 0:
                            logger.debug(f"Price for {symbol} obtained: {price_dec}")
                            return price_dec
                    except (InvalidOperation, TypeError):
                        continue

            logger.warning(
                f"No valid price (last, close, bid, ask, mid) found in ticker for {symbol} on attempt {attempts + 1}. Ticker: {ticker}"
            )
            raise ccxt_async.ExchangeError("No valid price found in ticker data.")
        except Exception as e:
            if not await _handle_fetch_exception(e, logger, attempts, total_attempts, f"price for {symbol}", symbol):
                return None
        attempts += 1
    logger.error(f"Failed to fetch price for {symbol} after {total_attempts} attempts.")
    return None


async def fetch_klines_ccxt(
    exchange: ccxt_async.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 250,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:  # Return Optional[pd.DataFrame] to align with potential empty return
    import pandas as pd  # Ensure pandas is imported locally if not globally

    current_logger = logger or module_logger
    if not exchange.has["fetchOHLCV"]:
        current_logger.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()  # Return empty DataFrame on failure

    total_attempts = MAX_API_RETRIES + 1
    for attempt in range(total_attempts):
        try:
            current_logger.debug(
                f"Fetching klines for {symbol} (Timeframe: {timeframe}, Limit: {limit}) (Attempt {attempt + 1}/{total_attempts})"
            )
            ohlcv_data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if (
                ohlcv_data
                and isinstance(ohlcv_data, list)
                and len(ohlcv_data) > 0
                and all(isinstance(row, list) and len(row) >= 6 for row in ohlcv_data)
            ):
                df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
                df.dropna(subset=["timestamp"], inplace=True)
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
                df.set_index("timestamp", inplace=True)

                for col in ["open", "high", "low", "close", "volume"]:
                    try:
                        df[col] = (
                            df[col]
                            .astype(object)
                            .apply(lambda x: Decimal(str(x)) if pd.notna(x) and str(x).strip() != "" else None)
                        )
                    except (InvalidOperation, TypeError) as conv_err:
                        current_logger.warning(
                            f"Could not convert column '{col}' to Decimal for {symbol} due to: {conv_err}. "
                            f"Falling back to pd.to_numeric, data might lose precision or be invalid."
                        )
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                df.dropna(subset=["open", "high", "low", "close"], how="any", inplace=True)
                df = df[
                    df["close"].apply(lambda x: isinstance(x, Decimal) and x > Decimal(0) or (pd.notna(x) and x > 0))
                ].copy()
                df = df[
                    df["volume"].apply(lambda x: isinstance(x, Decimal) and x >= Decimal(0) or (pd.notna(x) and x >= 0))
                ].copy()

                if df.empty:
                    current_logger.warning(
                        f"Klines data for {symbol} {timeframe} is empty after cleaning and validation."
                    )
                    raise ccxt_async.ExchangeError("Cleaned kline data is empty.")

                df.sort_index(inplace=True)
                current_logger.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}.")
                return df
            else:
                current_logger.warning(
                    f"Received empty or invalid kline data structure for {symbol} {timeframe}. "
                    f"Data: {ohlcv_data[:2] if ohlcv_data else 'None'}... (Attempt {attempt + 1})"
                )
                raise ccxt_async.ExchangeError("Empty or invalid kline data structure from exchange.")
        except Exception as e:
            if not await _handle_fetch_exception(
                e, current_logger, attempt, total_attempts, f"klines for {symbol} {timeframe}", symbol
            ):
                return pd.DataFrame()  # Return empty DataFrame on non-retryable error

    current_logger.error(f"Failed to fetch klines for {symbol} {timeframe} after {total_attempts} attempts.")
    return pd.DataFrame()  # Return empty DataFrame after all attempts fail


async def fetch_orderbook_ccxt(
    exchange: ccxt_async.Exchange, symbol: str, limit: int, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    if not exchange.has["fetchOrderBook"]:
        logger.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    attempts = 0
    total_attempts = MAX_API_RETRIES + 1
    while attempts < total_attempts:
        try:
            logger.debug(f"Fetching order book for {symbol} (Limit: {limit}) (Attempt {attempts + 1}/{total_attempts})")
            order_book = await exchange.fetch_order_book(symbol, limit=limit)

            if (
                order_book
                and isinstance(order_book, dict)
                and "bids" in order_book
                and isinstance(order_book["bids"], list)
                and "asks" in order_book
                and isinstance(order_book["asks"], list)
            ):
                if not order_book["bids"] and not order_book["asks"]:
                    logger.warning(f"Order book for {symbol} fetched but bids and asks arrays are empty.")
                return order_book
            else:
                logger.warning(
                    f"Invalid order book structure received for {symbol} on attempt {attempts + 1}. "
                    f"Data: {str(order_book)[:200]}..."
                )
                raise ccxt_async.ExchangeError("Invalid order book structure received.")
        except Exception as e:
            if not await _handle_fetch_exception(
                e, logger, attempts, total_attempts, f"orderbook for {symbol}", symbol
            ):
                return None
        attempts += 1
    logger.error(f"Failed to fetch order book for {symbol} after {total_attempts} attempts.")
    return None


async def fetch_balance(
    exchange: ccxt_async.Exchange, currency: str, logger: logging.Logger, params: Optional[Dict] = None
) -> Optional[Decimal]:
    request_params = params.copy() if params is not None else {}

    if exchange.id == "bybit" and "accountType" not in request_params:
        default_type = exchange.options.get("defaultType", "").upper()
        if default_type == "UNIFIED":
            request_params["accountType"] = "UNIFIED"
        elif default_type in ["LINEAR", "INVERSE", "CONTRACT"]:
            request_params["accountType"] = "CONTRACT"
        elif default_type == "SPOT":
            request_params["accountType"] = "SPOT"

    attempts = 0
    total_attempts = MAX_API_RETRIES + 1
    while attempts < total_attempts:
        try:
            logger.debug(
                f"Fetching balance for {currency} (Attempt {attempts + 1}/{total_attempts}). "
                f"Params: {request_params if request_params else 'None'}"
            )
            balance_info = await exchange.fetch_balance(params=request_params)

            if balance_info:
                currency_data = balance_info.get(currency.upper())
                available_balance_str = None

                if currency_data and currency_data.get("free") is not None:
                    available_balance_str = str(currency_data["free"])
                elif currency_data and currency_data.get("total") is not None:
                    available_balance_str = str(currency_data["total"])
                    logger.warning(
                        f"Using 'total' balance for {currency} as 'free' is unavailable. "
                        f"This might include locked funds."
                    )
                elif (
                    "free" in balance_info
                    and isinstance(balance_info["free"], dict)
                    and balance_info["free"].get(currency.upper()) is not None
                ):
                    available_balance_str = str(balance_info["free"][currency.upper()])

                if available_balance_str is not None:
                    try:
                        final_balance = Decimal(available_balance_str)
                        if final_balance >= Decimal(0):
                            logger.info(f"Available {currency} balance: {final_balance:.8f}")
                            return final_balance
                        else:
                            logger.error(
                                f"Parsed balance for {currency} is negative ({final_balance}). This is unusual."
                            )
                    except InvalidOperation:
                        logger.error(
                            f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}."
                        )
                else:
                    logger.error(
                        f"Could not determine free balance for {currency}. "
                        f"Relevant balance keys: {list(balance_info.keys() if isinstance(balance_info, dict) else [])}. "
                        f"Currency data: {currency_data}"
                    )
            else:
                logger.error(f"Balance info response was None or empty on attempt {attempts + 1}.")

            raise ccxt_async.ExchangeError(f"Balance parsing or fetch failed for {currency}.")
        except Exception as e:
            if not await _handle_fetch_exception(
                e, logger, attempts, total_attempts, f"balance for {currency}", currency
            ):
                return None
        attempts += 1
    logger.error(f"Failed to fetch balance for {currency} after {total_attempts} attempts.")
    return None


async def get_market_info(
    exchange: ccxt_async.Exchange, symbol: str, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    cache_key = f"{exchange.id}:{symbol}"
    if cache_key in _market_info_cache:
        logger.debug(f"Using cached market info for {symbol}.")
        return _market_info_cache[cache_key]

    try:
        if not exchange.markets or symbol not in exchange.markets:
            logger.info(f"Market info for {symbol} not found or markets not loaded. Reloading markets...")
            await exchange.load_markets(reload=True)

        if symbol not in exchange.markets:
            logger.error(f"Market {symbol} still not found after reloading markets.")
            return None

        market = exchange.market(symbol)
        if not market:
            logger.error(f"exchange.market({symbol}) returned None despite symbol being in markets list.")
            return None

        market.setdefault("precision", {})
        market["precision"].setdefault("price", "1e-8")
        market["precision"].setdefault("amount", "1e-8")

        market.setdefault("limits", {})
        market["limits"].setdefault("amount", {}).setdefault("min", "0")
        market["limits"].setdefault("cost", {}).setdefault("min", "0")

        market["is_contract"] = market.get("contract", False) or market.get("type", "unknown").lower() in [
            "swap",
            "future",
            "option",
            "linear",
            "inverse",
        ]

        if "amountPrecision" not in market or not isinstance(market.get("amountPrecision"), int):
            amount_step_val = market["precision"].get("amount")
            derived_precision = 8
            if isinstance(amount_step_val, (int)) and amount_step_val >= 0:
                derived_precision = amount_step_val
            elif isinstance(amount_step_val, (float, str, Decimal)):
                try:
                    step = Decimal(str(amount_step_val))
                    if step > 0:
                        derived_precision = abs(step.normalize().as_tuple().exponent)
                except (InvalidOperation, TypeError):
                    logger.warning(
                        f"Could not derive amountPrecision from step '{amount_step_val}' for {symbol}. Using default."
                    )
            market["amountPrecision"] = derived_precision

        logger.debug(
            f"Market Info for {symbol}: Type={market.get('type')}, Contract={market['is_contract']}, "
            f"TickSize(PriceStep)={market['precision']['price']}, AmountStep={market['precision']['amount']}, "
            f"AmountPrecision(DecimalPlaces)={market['amountPrecision']}"
        )
        _market_info_cache[cache_key] = market
        return market
    except Exception as e:
        logger.error(f"Error getting or processing market info for {symbol}: {e}", exc_info=True)
        return None


async def get_open_position(
    exchange: ccxt_async.Exchange, symbol: str, market_info: Dict[str, Any], logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    if not exchange.has.get("fetchPositions"):
        logger.warning(f"Exchange {exchange.id} does not support fetchPositions.")
        return None

    market_id = market_info.get("id")
    if not market_id:
        logger.error(f"Market ID missing in market_info for {symbol}. Cannot reliably fetch position.")
        return None

    positions: List[Dict[str, Any]] = []
    try:
        logger.debug(f"Fetching position for {symbol} (Market ID: {market_id})")
        fetched_positions_raw = await exchange.fetch_positions([symbol])
        positions = [p for p in fetched_positions_raw if p.get("symbol") == symbol]

    except ccxt_async.ArgumentsRequired:
        logger.debug(
            f"fetchPositions for {exchange.id} with symbol argument failed or is not supported. "
            f"Attempting to fetch all positions and filter for {symbol}."
        )
        try:
            all_positions = await exchange.fetch_positions()
            positions = [
                p
                for p in all_positions
                if p.get("symbol") == symbol or (p.get("info") and p["info"].get("symbol") == market_id)
            ]
        except Exception as e_all:
            logger.error(f"Error fetching all positions while trying to find {symbol}: {e_all}", exc_info=True)
            return None
    except ccxt_async.ExchangeError as e:
        no_pos_indicators = ["position not found", "no position", "position does not exist"]
        if any(msg in str(e).lower() for msg in no_pos_indicators) or (
            exchange.id == "bybit" and getattr(e, "code", None) in [110025, 10001]
        ):
            logger.info(f"No open position found for {symbol} (Exchange reported: {e}).")
            return None
        logger.error(f"Exchange error fetching positions for {symbol}: {e}.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching positions for {symbol}: {e}", exc_info=True)
        return None

    if not positions:
        logger.info(f"No position data structures returned or matched for {symbol}.")
        return None

    active_position_data = None
    raw_amount_step = market_info.get("precision", {}).get("amount", "1e-8")
    try:
        size_threshold = Decimal(str(raw_amount_step)) / Decimal("100")
        if size_threshold <= 0:
            size_threshold = Decimal("1e-9")
    except InvalidOperation:
        size_threshold = Decimal("1e-9")

    for pos_data in positions:
        size_str = (
            pos_data.get("contracts") or pos_data.get("info", {}).get("size") or pos_data.get("info", {}).get("qty")
        )
        if size_str is None:
            continue

        try:
            pos_size_dec = Decimal(str(size_str))
            bybit_v5_pos_side = pos_data.get("info", {}).get("positionSide", "").lower()
            if exchange.id == "bybit" and bybit_v5_pos_side == "none" and abs(pos_size_dec) <= size_threshold:
                continue

            if abs(pos_size_dec) > size_threshold:
                active_position_data = pos_data.copy()
                active_position_data["contractsDecimal"] = abs(pos_size_dec)
                current_side = active_position_data.get("side", "").lower()
                if not current_side or current_side == "none":
                    if exchange.id == "bybit" and bybit_v5_pos_side in ["buy", "sell"]:
                        current_side = "long" if bybit_v5_pos_side == "buy" else "short"
                    elif pos_size_dec > size_threshold:
                        current_side = "long"
                    elif pos_size_dec < -size_threshold:
                        current_side = "short"
                    else:
                        continue
                active_position_data["side"] = current_side
                ep_str = active_position_data.get("entryPrice") or active_position_data.get("info", {}).get("avgPrice")
                active_position_data["entryPriceDecimal"] = Decimal(str(ep_str)) if ep_str is not None else None

                field_map = {
                    "markPriceDecimal": ["markPrice"],
                    "liquidationPriceDecimal": ["liquidationPrice", "liqPrice"],
                    "unrealizedPnlDecimal": ["unrealizedPnl", "unrealisedPnl", "pnl", ("info", "unrealisedPnl")],
                    "stopLossPriceDecimal": ["stopLoss", "stopLossPrice", "slPrice", ("info", "stopLoss")],
                    "takeProfitPriceDecimal": ["takeProfit", "takeProfitPrice", "tpPrice", ("info", "takeProfit")],
                    "trailingStopLossValue": [
                        ("info", "trailingStop"),
                        ("info", "trailing_stop"),
                        ("info", "tpslTriggerPrice"),
                    ],
                    "trailingStopActivationPrice": [
                        ("info", "activePrice"),
                        ("info", "triggerPrice"),
                        ("info", "trailing_trigger_price"),
                    ],
                }
                for dec_key, str_keys_list in field_map.items():
                    val_str = None
                    for sk_item in str_keys_list:
                        if isinstance(sk_item, tuple):
                            val_str = active_position_data.get(sk_item[0], {}).get(sk_item[1])
                        else:
                            val_str = active_position_data.get(sk_item)
                        if val_str is not None:
                            break
                    if val_str is not None and str(val_str).strip():
                        if str(val_str) == "0" and dec_key in ["stopLossPriceDecimal", "takeProfitPriceDecimal"]:
                            active_position_data[dec_key] = None
                        else:
                            try:
                                active_position_data[dec_key] = Decimal(str(val_str))
                            except (InvalidOperation, TypeError):
                                active_position_data[dec_key] = None
                    else:
                        active_position_data[dec_key] = None
                ts_str = (
                    active_position_data.get("timestamp")
                    or active_position_data.get("info", {}).get("updatedTime")
                    or active_position_data.get("info", {}).get("updated_at")
                    or active_position_data.get("info", {}).get("createTime")
                )
                active_position_data["timestamp_ms"] = int(float(ts_str)) if ts_str else None
                break
        except (InvalidOperation, ValueError, TypeError) as e:
            logger.warning(f"Error parsing position data for {symbol}: {e}. Data: {pos_data}", exc_info=True)
            continue

    if active_position_data:
        logger.info(
            f"Active {active_position_data.get('side', 'N/A').upper()} position found for {symbol}: "
            f"Size={active_position_data.get('contractsDecimal', 'N/A')}, "
            f"Entry={active_position_data.get('entryPriceDecimal', 'N/A')}"
        )
        return active_position_data

    logger.info(f"No active open position found for {symbol} after filtering (size > {size_threshold:.8f}).")
    return None


async def set_leverage_ccxt(
    exchange: ccxt_async.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger
) -> bool:
    if not market_info.get("is_contract", False):
        logger.info(f"Leverage setting skipped for {symbol} as it's not a contract market.")
        return True

    if not (isinstance(leverage, int) and leverage > 0):
        logger.warning(f"Invalid leverage value {leverage} for {symbol}. Must be a positive integer.")
        return False

    if not (hasattr(exchange, "set_leverage") and callable(exchange.set_leverage)):
        logger.error(f"Exchange {exchange.id} does not support set_leverage method via CCXT.")
        return False

    logger.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
    params = {}

    try:
        response = await exchange.set_leverage(leverage, symbol, params=params)
        logger.debug(f"Set leverage response for {symbol}: {response}")

        if exchange.id == "bybit" and isinstance(response, dict):
            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "").lower()
            if ret_code == 0:
                logger.info(f"{NEON_GREEN}Leverage for {symbol} successfully set to {leverage}x (Bybit).{RESET}")
                return True
            elif ret_code == 110043 or "leverage not modified" in ret_msg or "same leverage" in ret_msg:
                logger.info(f"Leverage for {symbol} was already {leverage}x (Bybit: {ret_code} - {ret_msg}).")
                return True
            else:
                logger.error(f"Bybit error setting leverage for {symbol}: {ret_msg} (Code: {ret_code})")
                return False

        logger.info(f"{NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Generic CCXT response).{RESET}")
        return True
    except ccxt_async.ExchangeError as e:
        err_str, code = str(e).lower(), getattr(e, "code", None)
        if "leverage not modified" in err_str or "no change" in err_str or (exchange.id == "bybit" and code == 110043):
            logger.info(f"Leverage for {symbol} already {leverage}x (Confirmed by error: {e}).")
            return True
        logger.error(f"Exchange error setting leverage for {symbol} to {leverage}x: {e} (Code: {code})")
    except Exception as e:
        logger.error(f"Unexpected error setting leverage for {symbol} to {leverage}x: {e}", exc_info=True)
    return False


async def place_trade(
    exchange: ccxt_async.Exchange,
    symbol: str,
    trade_signal: str,
    position_size: Decimal,
    market_info: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    order_type: str = "market",
    limit_price: Optional[Decimal] = None,
    reduce_only: bool = False,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    current_logger = logger or module_logger
    side = "buy" if trade_signal.upper() == "BUY" else "sell"
    action_description = "Reduce-Only" if reduce_only else "Open/Increase"

    try:
        if not (isinstance(position_size, Decimal) and position_size > 0):
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Invalid position_size ({position_size}). Must be a positive Decimal."
            )
            return None

        amount_str_for_api = exchange.amount_to_precision(symbol, float(position_size))
        amount_for_api = float(amount_str_for_api)

        if amount_for_api <= 0:
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Position size after formatting ({amount_for_api}) is not positive."
            )
            return None
    except Exception as e:
        current_logger.error(
            f"Trade aborted for {symbol} ({side}): Error formatting position_size {position_size}: {e}", exc_info=True
        )
        return None

    price_for_api: Optional[float] = None
    price_log_str: Optional[str] = None
    if order_type.lower() == "limit":
        if not (isinstance(limit_price, Decimal) and limit_price > 0):
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Limit order chosen, but invalid limit_price ({limit_price})."
            )
            return None
        try:
            price_log_str = exchange.price_to_precision(symbol, float(limit_price))
            price_for_api = float(price_log_str)
            if price_for_api <= 0:
                raise ValueError("Formatted limit price is not positive.")
        except Exception as e:
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Error formatting limit_price {limit_price}: {e}", exc_info=True
            )
            return None
    elif order_type.lower() != "market":
        current_logger.error(f"Unsupported order type '{order_type}' for {symbol}. Only 'market' or 'limit' supported.")
        return None

    final_params = {"reduceOnly": reduce_only}
    if exchange.id == "bybit":
        final_params["positionIdx"] = 0

    if params:
        final_params.update(params)

    if reduce_only and order_type.lower() == "market" and "timeInForce" not in final_params:
        final_params["timeInForce"] = "IOC"

    base_currency = market_info.get("base", "units")
    log_message = (
        f"Placing {action_description} {side.upper()} {order_type.upper()} order for {symbol}: "
        f"Size = {amount_for_api} {base_currency}"
    )
    if price_log_str:
        log_message += f", Price = {price_log_str}"
    log_message += f", Params = {final_params}"
    current_logger.info(log_message)

    try:
        order = await exchange.create_order(
            symbol, order_type.lower(), side, amount_for_api, price_for_api, final_params
        )
        if order:
            current_logger.info(
                f"{NEON_GREEN}{action_description} order for {symbol} PLACED successfully. "
                f"ID: {order.get('id')}, Status: {order.get('status', 'N/A')}{RESET}"
            )
            return order
        else:
            current_logger.error(f"Order placement for {symbol} returned None without raising an exception.")
            return None
    except ccxt_async.InsufficientFunds as e:
        current_logger.error(
            f"{NEON_RED}Insufficient funds to place {side} {order_type} order for {symbol}: {e}{RESET}"
        )
    except ccxt_async.InvalidOrder as e:
        current_logger.error(
            f"{NEON_RED}Invalid order parameters for {symbol} ({side}, {order_type}): {e}. "
            f"Details: Amount={amount_for_api}, Price={price_for_api}, Params={final_params}{RESET}",
            exc_info=True,
        )
    except ccxt_async.ExchangeError as e:
        current_logger.error(
            f"{NEON_RED}Exchange error placing {action_description} order for {symbol}: {e}{RESET}", exc_info=True
        )
    except Exception as e:
        current_logger.error(
            f"{NEON_RED}Unexpected error placing {action_description} order for {symbol}: {e}{RESET}", exc_info=True
        )
    return None


async def _set_position_protection(
    exchange: ccxt_async.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Union[Decimal, str]] = None,
    tsl_activation_price: Optional[Union[Decimal, str]] = None,
) -> bool:
    if "bybit" not in exchange.id.lower():
        logger.error("Position protection logic (_set_position_protection) is currently Bybit V5 specific.")
        return False
    if not market_info.get("is_contract", False):
        logger.warning(f"Protection skipped for {symbol}: not a contract market.")
        return True

    if not position_info or "side" not in position_info or not position_info["side"]:
        logger.error(f"Cannot set protection for {symbol}: invalid or missing position_info (especially 'side').")
        return False

    pos_side_str = position_info["side"].lower()
    pos_idx_raw = position_info.get("info", {}).get("positionIdx", 0)
    try:
        position_idx = int(pos_idx_raw)
    except (ValueError, TypeError):
        logger.warning(f"Invalid positionIdx '{pos_idx_raw}' for {symbol}, defaulting to 0.")
        position_idx = 0

    market_type = market_info.get("type", "").lower()
    if market_info.get("linear", False) or market_type == "linear":
        category = "linear"
    elif market_info.get("inverse", False) or market_type == "inverse":
        category = "inverse"
    elif market_info.get("spot", False) or market_type == "spot":
        category = "spot"
        logger.warning(
            f"Attempting to set protection on SPOT symbol {symbol}, this may not be fully supported by Bybit's position protection endpoint."
        )
    else:
        category = "linear"
        logger.warning(
            f"Market category for {symbol} is ambiguous (type: {market_type}). Defaulting to 'linear' for protection API call."
        )

    api_params: Dict[str, Any] = {
        "category": category,
        "symbol": market_info["id"],
        "positionIdx": position_idx,
    }
    log_parts = [
        f"Attempting to set/update protection for {symbol} ({pos_side_str.upper()}, PosIdx:{position_idx}, Cat:{category}):"
    ]
    protection_fields_to_send: Dict[str, str] = {}

    try:
        price_precision_places = get_price_precision(market_info, logger)
        min_tick_size_dec = get_min_tick_size(market_info, logger)
        if not (min_tick_size_dec and min_tick_size_dec > 0):
            min_tick_size_dec = Decimal(f"1e-{price_precision_places}")

        def format_price_for_api(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not (price_decimal and isinstance(price_decimal, Decimal) and price_decimal > 0):
                return None
            return exchange.price_to_precision(symbol, float(price_decimal))

        if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0:
            distance_precision_places = abs(min_tick_size_dec.normalize().as_tuple().exponent)
            tsl_dist_str = exchange.decimal_to_precision(
                trailing_stop_distance,
                ccxt_async.ROUND,  # type: ignore
                distance_precision_places,
                ccxt_async.DECIMAL_PLACES,  # type: ignore
                ccxt_async.NO_PADDING,  # type: ignore
            )
            if Decimal(tsl_dist_str) < min_tick_size_dec:
                tsl_dist_str = str(min_tick_size_dec.quantize(Decimal(f"1e-{distance_precision_places}")))

            tsl_act_price_str_final: Optional[str] = None
            if tsl_activation_price == "0":
                tsl_act_price_str_final = "0"
            elif isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0:
                tsl_act_price_str_final = format_price_for_api(tsl_activation_price)

            if tsl_dist_str and Decimal(tsl_dist_str) > 0 and tsl_act_price_str_final is not None:
                protection_fields_to_send.update({"trailingStop": tsl_dist_str, "activePrice": tsl_act_price_str_final})
                log_parts.append(
                    f"  - Trailing Stop: Distance={tsl_dist_str}, ActivationPrice={tsl_act_price_str_final}"
                )
            else:
                logger.error(
                    f"Failed to format TSL parameters for {symbol}. "
                    f"DistanceInput='{trailing_stop_distance}', FormattedDist='{tsl_dist_str}', "
                    f"ActivationInput='{tsl_activation_price}', FormattedAct='{tsl_act_price_str_final}'"
                )
        elif trailing_stop_distance == "0":
            protection_fields_to_send["trailingStop"] = "0"
            log_parts.append("  - Trailing Stop: Removing (distance set to '0')")

        if stop_loss_price is not None:
            sl_price_str = "0" if stop_loss_price == Decimal(0) else format_price_for_api(stop_loss_price)
            if sl_price_str is not None:
                protection_fields_to_send["stopLoss"] = sl_price_str
                log_parts.append(f"  - Fixed Stop Loss: {sl_price_str}")

        if take_profit_price is not None:
            tp_price_str = "0" if take_profit_price == Decimal(0) else format_price_for_api(take_profit_price)
            if tp_price_str is not None:
                protection_fields_to_send["takeProfit"] = tp_price_str
                log_parts.append(f"  - Fixed Take Profit: {tp_price_str}")

    except Exception as fmt_err:
        logger.error(f"Error formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
        return False

    if not protection_fields_to_send:
        logger.info(f"No valid protection parameters to set or update for {symbol}.")
        return True

    api_params.update(protection_fields_to_send)
    logger.info("\n".join(log_parts))
    logger.debug(f"  API Call to set trading stop/protection for {symbol}: params={api_params}")

    try:
        method_name_camel = "v5PrivatePostPositionSetTradingStop"
        method_name_snake = "v5_private_post_position_set_trading_stop"

        if hasattr(exchange, method_name_camel):
            set_protection_method = getattr(exchange, method_name_camel)
        elif hasattr(exchange, method_name_snake):
            set_protection_method = getattr(exchange, method_name_snake)
        else:
            logger.error(
                f"CCXT instance for {exchange.id} is missing the required method for setting position protection "
                f"(checked for '{method_name_camel}' and '{method_name_snake}'). "
                f"Ensure CCXT library is up-to-date and supports Bybit V5 position protection."
            )
            return False

        response = await set_protection_method(api_params)
        logger.debug(f"Set protection raw API response for {symbol}: {response}")

        if isinstance(response, dict) and response.get("retCode") == 0:
            logger.info(
                f"{NEON_GREEN}Protection for {symbol} successfully set/updated. "
                f"Message: {response.get('retMsg', 'OK')}{RESET}"
            )
            return True
        else:
            logger.error(
                f"{NEON_RED}Failed to set protection for {symbol}. "
                f"API Response Code: {response.get('retCode')}, Message: {response.get('retMsg')}{RESET}"
            )
            return False
    except Exception as e:
        logger.error(f"{NEON_RED}Error during API call to set protection for {symbol}: {e}{RESET}", exc_info=True)
        return False


async def set_trailing_stop_loss(
    exchange: ccxt_async.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None,
) -> bool:
    if not config.get("enable_trailing_stop", False):
        logger.info(f"Trailing Stop Loss is disabled in config for {symbol}.")
        if take_profit_price and isinstance(take_profit_price, Decimal) and take_profit_price > 0:
            logger.info(f"TSL disabled, but attempting to set provided Take Profit for {symbol}.")
            return await _set_position_protection(
                exchange, symbol, market_info, position_info, logger, take_profit_price=take_profit_price
            )
        return True

    if not market_info.get("is_contract", False):
        logger.warning(f"Trailing Stop Loss is typically for contract markets. Skipped for {symbol} (not a contract).")
        if take_profit_price and isinstance(take_profit_price, Decimal) and take_profit_price > 0:
            logger.info(f"Market is not a contract, but attempting to set provided Take Profit for {symbol}.")
            return await _set_position_protection(
                exchange, symbol, market_info, position_info, logger, take_profit_price=take_profit_price
            )
        return True

    try:
        callback_rate_str = str(config.get("trailing_stop_callback_rate", "0.005"))
        activation_percentage_str = str(config.get("trailing_stop_activation_percentage", "0.003"))
        callback_rate = Decimal(callback_rate_str)
        activation_percentage = Decimal(activation_percentage_str)
        if callback_rate <= 0:
            raise ValueError("Trailing stop callback rate must be positive.")
        if activation_percentage < 0:
            raise ValueError("Trailing stop activation percentage must be non-negative.")
    except (InvalidOperation, ValueError, TypeError) as e:
        logger.error(f"Invalid TSL parameters in configuration for {symbol}: {e}. Please check config.")
        return False

    try:
        entry_price = position_info.get("entryPriceDecimal")
        position_side = position_info.get("side", "").lower()
        if not (isinstance(entry_price, Decimal) and entry_price > 0):
            raise ValueError(f"Invalid or missing entry price in position_info: {entry_price}")
        if position_side not in ["long", "short"]:
            raise ValueError(f"Invalid or missing side in position_info: {position_side}")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid position information for TSL setup ({symbol}): {e}. Position: {position_info}")
        return False

    try:
        price_precision_places = get_price_precision(market_info, logger)
        min_tick_size = get_min_tick_size(market_info, logger)
        quantize_fallback_tick = Decimal(f"1e-{price_precision_places}")
        effective_tick_size = min_tick_size if min_tick_size and min_tick_size > 0 else quantize_fallback_tick
        if not (effective_tick_size > 0):
            logger.error(f"Could not determine a valid tick size for TSL calculations for {symbol}.")
            return False

        current_market_price = await fetch_current_price_ccxt(exchange, symbol, logger)
        if not current_market_price:
            logger.warning(
                f"Could not fetch current market price for {symbol} for TSL logic. Using entry price as reference."
            )
            current_market_price = entry_price

        price_change_for_activation = entry_price * activation_percentage
        raw_activation_price = entry_price + (
            price_change_for_activation if position_side == "long" else -price_change_for_activation
        )

        activate_immediately = False
        if config.get("tsl_activate_immediately_if_profitable", True):
            if position_side == "long" and current_market_price >= raw_activation_price:
                activate_immediately = True
            elif position_side == "short" and current_market_price <= raw_activation_price:
                activate_immediately = True

        final_activation_price_param: Union[Decimal, str]
        calculated_activation_price_for_log: Optional[Decimal] = None

        if activate_immediately:
            final_activation_price_param = "0"
            calculated_activation_price_for_log = current_market_price
            logger.info(
                f"TSL for {symbol} ({position_side}): Position is already profitable beyond activation point. "
                f"Setting activePrice='0' for immediate trailing based on current market price ({current_market_price})."
            )
        else:
            min_profit_activation_price = (
                entry_price + effective_tick_size if position_side == "long" else entry_price - effective_tick_size
            )
            if position_side == "long":
                if raw_activation_price < min_profit_activation_price:
                    raw_activation_price = min_profit_activation_price
                if raw_activation_price < current_market_price:  # type: ignore
                    raw_activation_price = current_market_price + effective_tick_size  # type: ignore
                rounding_mode = ROUND_UP
            else:  # short
                if raw_activation_price > min_profit_activation_price:
                    raw_activation_price = min_profit_activation_price
                if raw_activation_price > current_market_price:  # type: ignore
                    raw_activation_price = current_market_price - effective_tick_size  # type: ignore
                rounding_mode = ROUND_DOWN

            calculated_activation_price = (raw_activation_price / effective_tick_size).quantize(
                Decimal("1"), rounding=rounding_mode
            ) * effective_tick_size

            if calculated_activation_price <= 0:
                logger.error(
                    f"Calculated TSL Activation Price ({calculated_activation_price}) is not positive for {symbol}. Cannot set TSL."
                )
                return False
            if position_side == "long" and calculated_activation_price <= entry_price:
                logger.warning(
                    f"Calculated TSL Activation Price ({calculated_activation_price}) for LONG {symbol} is not profitable vs Entry ({entry_price}). "
                    f"Adjusting to one tick above entry."
                )
                calculated_activation_price = ((entry_price + effective_tick_size) / effective_tick_size).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * effective_tick_size
            elif position_side == "short" and calculated_activation_price >= entry_price:
                logger.warning(
                    f"Calculated TSL Activation Price ({calculated_activation_price}) for SHORT {symbol} is not profitable vs Entry ({entry_price}). "
                    f"Adjusting to one tick below entry."
                )
                calculated_activation_price = ((entry_price - effective_tick_size) / effective_tick_size).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * effective_tick_size

            if calculated_activation_price <= 0:
                logger.error(
                    f"Final TSL Activation Price ({calculated_activation_price}) non-positive for {symbol}. Cannot set TSL."
                )
                return False
            final_activation_price_param = calculated_activation_price
            calculated_activation_price_for_log = calculated_activation_price

        raw_trail_distance = entry_price * callback_rate
        trail_distance = (raw_trail_distance / effective_tick_size).quantize(
            Decimal("1"), rounding=ROUND_UP
        ) * effective_tick_size
        if trail_distance < effective_tick_size:
            trail_distance = effective_tick_size
        if trail_distance <= 0:
            logger.error(
                f"Calculated TSL trail distance ({trail_distance}) is not positive for {symbol}. Cannot set TSL."
            )
            return False

        log_act_price_str = (
            f"{calculated_activation_price_for_log:.{price_precision_places}f}"
            if calculated_activation_price_for_log
            else "N/A (Immediate)"
        )
        logger.info(
            f"Calculated TSL parameters for {symbol} ({position_side.upper()}):\n"
            f"  Entry Price: {entry_price:.{price_precision_places}f}\n"
            f"  Activation Price (for API): '{final_activation_price_param}' (Based on calculated: {log_act_price_str}, from {activation_percentage:.2%})\n"
            f"  Trail Distance: {trail_distance:.{price_precision_places}f} (From callback rate: {callback_rate:.2%})"
        )
        if take_profit_price and isinstance(take_profit_price, Decimal) and take_profit_price > 0:
            logger.info(f"  Also setting Take Profit at: {take_profit_price:.{price_precision_places}f}")

        return await _set_position_protection(
            exchange,
            symbol,
            market_info,
            position_info,
            logger,
            stop_loss_price=None,
            take_profit_price=take_profit_price
            if isinstance(take_profit_price, Decimal) and take_profit_price > 0
            else None,
            trailing_stop_distance=trail_distance,
            tsl_activation_price=final_activation_price_param,
        )
    except Exception as e:
        logger.error(f"Unexpected error during TSL setup for {symbol}: {e}", exc_info=True)
        return False
