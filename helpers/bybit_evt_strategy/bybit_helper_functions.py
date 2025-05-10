# --- START OF FILE bybit_helper_functions.py ---

# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bybit V5 CCXT Helper Functions (v2.9 - Add cancel/fetch order, Fix Ticker Validation)

This module provides a collection of robust, reusable, and enhanced **synchronous**
helper functions designed for interacting with the Bybit exchange (V5 API)
using the CCXT library.

**Note:** This version is synchronous. For asynchronous operations, use
`ccxt.async_support` and adapt these functions accordingly (e.g., using `await`).

Core Functionality Includes:
- Exchange Initialization: Securely sets up the ccxt.bybit exchange instance.
- Account Configuration: Set leverage, margin mode, position mode.
- Market Data Retrieval: Validated fetchers for tickers, OHLCV, order books, etc.
- Order Management: Place market, limit, stop orders; cancel orders; fetch orders.
- Position Management: Fetch positions, close positions.
- Balance & Margin: Fetch balances, calculate margin estimates.
- Utilities: Market validation.

Key Enhancements in v2.9:
- Added `cancel_order` and `fetch_order` helper functions.
- Fixed `ValueError` in `fetch_ticker_validated` when timestamp is missing.
- Improved exception message clarity in `fetch_ticker_validated`.
- Explicitly imports utilities from `bybit_utils`.

Dependencies:
- `logger`: Pre-configured `logging.Logger` object (from main script).
- `Config`: Configuration class/object (from main script or `config_models`).
- `bybit_utils.py`: Utility functions and retry decorator.
"""

# Standard Library Imports
import logging
import random  # Used in fetch_ohlcv_paginated retry delay jitter
import sys
import time
from decimal import Decimal, DivisionByZero, getcontext
from typing import Any, Literal

# Third-party Libraries
try:
    import ccxt
except ImportError:
    print(
        "Error: CCXT library not found. Please install it: pip install ccxt",
        file=sys.stderr,
    )
    sys.exit(1)
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    print(
        "Warning: pandas library not found. OHLCV data will be list.", file=sys.stderr
    )
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print(
        "Warning: colorama library not found. Logs will not be colored.",
        file=sys.stderr,
    )

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()  # type: ignore
    COLORAMA_AVAILABLE = False

# --- Import Utilities from bybit_utils ---
try:
    from bybit_utils import (
        analyze_order_book,  # Ensure analyze_order_book is defined in bybit_utils
        format_amount,
        format_order_id,
        format_price,
        retry_api_call,
        safe_decimal_conversion,
        send_sms_alert,
    )

    print("[bybit_helpers] Successfully imported utilities from bybit_utils.")
except ImportError as e:
    print(
        f"FATAL ERROR [bybit_helpers]: Failed to import required functions/decorator from bybit_utils.py: {e}",
        file=sys.stderr,
    )
    print(
        "Ensure bybit_utils.py is in the same directory or accessible via PYTHONPATH.",
        file=sys.stderr,
    )
    sys.exit(1)
except NameError as e:
    print(
        f"FATAL ERROR [bybit_helpers]: A required name is not defined in bybit_utils.py: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# Set Decimal context precision
getcontext().prec = 28

# --- Logger Placeholder ---
# Actual logger MUST be provided by importing script (e.g., main.py)
# This provides a fallback if the module is imported before logger setup.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default level if not configured
    logger.info(
        f"Placeholder logger initialized for {__name__}. Main logger setup expected."
    )

# --- Config Placeholder ---
# Actual config object MUST be provided by importing script or passed to functions
try:
    from config_models import AppConfig  # Use the main config model

    Config = AppConfig  # Use AppConfig as the type hint
except ImportError:
    logger.error(
        "Could not import AppConfig from config_models. Type hinting will use basic 'object'."
    )
    Config = object  # Fallback type hint


# --- Helper Function Implementations ---


def _get_v5_category(
    market: dict[str, Any],
) -> Literal["linear", "inverse", "spot", "option"] | None:
    """Internal helper to determine the Bybit V5 category from a market object."""
    func_name = "_get_v5_category"
    if not market:
        return None

    # Use the logic from bybit_helpers v3.5 (more robust)
    info = market.get("info", {})
    category_from_info = info.get("category")
    if category_from_info in ["linear", "inverse", "spot", "option"]:
        return category_from_info  # type: ignore

    if market.get("spot", False):
        return "spot"
    if market.get("option", False):
        return "option"
    if market.get("linear", False):
        return "linear"
    if market.get("inverse", False):
        return "inverse"

    market_type = market.get("type")
    symbol = market.get("symbol", "N/A")

    if market_type == "spot":
        return "spot"
    if market_type == "option":
        return "option"

    if market_type in ["swap", "future"]:
        contract_type = str(info.get("contractType", "")).lower()
        settle_coin = market.get("settle", "").upper()
        if contract_type == "linear":
            return "linear"
        if contract_type == "inverse":
            return "inverse"
        if settle_coin in ["USDT", "USDC"]:
            return "linear"
        if settle_coin and settle_coin == market.get("base", "").upper():
            return "inverse"
        logger.debug(f"[{func_name}] Ambiguous derivative {symbol}. Assuming 'linear'.")
        return "linear"

    logger.warning(
        f"[{func_name}] Could not determine V5 category for market: {symbol}, Type: {market_type}"
    )
    return None


# Snippet 1 / Function 1: Initialize Bybit Exchange
@retry_api_call(
    max_retries_override=3,
    initial_delay_override=2.0,
    error_message_prefix="Exchange Init Failed",
)
def initialize_bybit(config: Config) -> ccxt.bybit | None:
    """Initializes and validates the Bybit CCXT exchange instance using V5 API settings."""
    func_name = "initialize_bybit"
    api_conf = config.api_config  # Access nested config
    strat_conf = config.strategy_config

    mode_str = "Testnet" if api_conf.testnet_mode else "Mainnet"
    logger.info(
        f"{Fore.BLUE}[{func_name}] Initializing Bybit V5 ({mode_str}, Sync)...{Style.RESET_ALL}"
    )
    try:
        exchange_class = getattr(ccxt, api_conf.exchange_id)
        exchange = exchange_class(
            {
                "apiKey": api_conf.api_key,
                "secret": api_conf.api_secret,
                "enableRateLimit": True,
                "options": {
                    # 'defaultType': api_conf.expected_market_type, # Less critical for V5 if category used
                    "adjustForTimeDifference": True,
                    "recvWindow": api_conf.default_recv_window,
                    "brokerId": f"PB_{strat_conf.name[:10].replace(' ', '_')}",  # Example broker ID
                },
            }
        )
        if api_conf.testnet_mode:
            logger.info(f"[{func_name}] Enabling Bybit Sandbox (Testnet) mode.")
            exchange.set_sandbox_mode(True)

        logger.debug(f"[{func_name}] Loading markets...")
        exchange.load_markets(reload=True)  # Synchronous load
        if not exchange.markets:
            raise ccxt.ExchangeError(f"[{func_name}] Failed to load markets.")
        logger.debug(
            f"[{func_name}] Markets loaded successfully ({len(exchange.markets)} symbols)."
        )

        # --- Authentication Check ---
        if api_conf.api_key and api_conf.api_secret:
            logger.debug(
                f"[{func_name}] Performing initial balance fetch for validation..."
            )
            # Use fetch_usdt_balance which specifies UNIFIED account
            balance_info = fetch_usdt_balance(exchange, config)  # Pass the main config
            if balance_info is None:
                # Error logged by fetch_usdt_balance, raise specific error here
                raise ccxt.AuthenticationError(
                    "Initial balance check failed. Verify API keys and permissions."
                )
            logger.info(f"[{func_name}] Initial balance check successful.")
        else:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] API keys not provided. Skipping auth check.{Style.RESET_ALL}"
            )

        # --- Optional: Set initial margin mode for the primary symbol ---
        try:
            market = exchange.market(api_conf.symbol)
            category = _get_v5_category(market)
            if category and category in ["linear", "inverse"]:
                logger.debug(
                    f"[{func_name}] Attempting to set initial margin mode '{strat_conf.default_margin_mode}' for {api_conf.symbol} (Cat: {category})..."
                )
                # Note: set_margin_mode might require specific account types or permissions
                exchange.set_margin_mode(
                    marginMode=strat_conf.default_margin_mode,
                    symbol=api_conf.symbol,
                    params={
                        "category": category,
                        "leverage": strat_conf.leverage,
                    },  # Can set leverage here too
                )
                logger.info(
                    f"[{func_name}] Initial margin mode potentially set to '{strat_conf.default_margin_mode}' for {api_conf.symbol}."
                )
            else:
                logger.warning(
                    f"[{func_name}] Cannot determine contract category for {api_conf.symbol}. Skipping initial margin mode set."
                )
        except (
            ccxt.NotSupported,
            ccxt.ExchangeError,
            ccxt.ArgumentsRequired,
            ccxt.BadSymbol,
        ) as e_margin:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Could not set initial margin mode for {api_conf.symbol}: {e_margin}. Verify account settings/permissions.{Style.RESET_ALL}"
            )

        logger.success(
            f"{Fore.GREEN}[{func_name}] Bybit exchange initialized successfully. Testnet: {api_conf.testnet_mode}.{Style.RESET_ALL}"
        )
        return exchange

    except (
        ccxt.AuthenticationError,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.ExchangeError,
    ) as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Initialization attempt failed: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        # Decorator handles retries for NetworkError etc., but AuthError is usually fatal here.
        if isinstance(e, ccxt.AuthenticationError):
            send_sms_alert(
                f"[BybitHelper] CRITICAL: Bybit Auth failed! {type(e).__name__}",
                config.sms_config,
            )
        raise  # Re-raise to be caught by caller or retry decorator
    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error during Bybit initialization: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[BybitHelper] CRITICAL: Bybit init failed! Unexpected: {type(e).__name__}",
            config.sms_config,
        )
        return None  # Return None on unexpected critical failure


# Snippet 2 / Function 2: Set Leverage
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def set_leverage(
    exchange: ccxt.bybit, symbol: str, leverage: int, config: Config
) -> bool:
    """Sets the leverage for a specific symbol on Bybit V5 (Linear/Inverse)."""
    func_name = "set_leverage"
    log_prefix = f"[{func_name}({symbol} -> {leverage}x)]"
    logger.info(f"{Fore.CYAN}{log_prefix} Setting leverage...{Style.RESET_ALL}")
    if leverage <= 0:
        logger.error(
            f"{Fore.RED}{log_prefix} Leverage must be positive: {leverage}{Style.RESET_ALL}"
        )
        return False
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Invalid market type for leverage: {symbol} ({category}).{Style.RESET_ALL}"
            )
            return False

        # Basic leverage validation against market limits (if available)
        try:
            limits = market.get("limits", {}).get("leverage", {})
            max_lev_str = limits.get("max")
            min_lev_str = limits.get("min", "1")
            max_lev = int(
                safe_decimal_conversion(max_lev_str, Decimal("100"))
            )  # Default max 100 if missing
            min_lev = int(
                safe_decimal_conversion(min_lev_str, Decimal("1"))
            )  # Default min 1
            if not (min_lev <= leverage <= max_lev):
                logger.error(
                    f"{Fore.RED}{log_prefix} Invalid leverage {leverage}x. Allowed: {min_lev}x - {max_lev}x.{Style.RESET_ALL}"
                )
                return False
        except Exception as e_lim:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Could not validate leverage limits: {e_lim}. Proceeding.{Style.RESET_ALL}"
            )

        params = {
            "category": category,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        logger.debug(f"{log_prefix} Calling exchange.set_leverage with params={params}")
        # CCXT's set_leverage for Bybit V5 calls POST /v5/position/set-leverage
        exchange.set_leverage(leverage, symbol, params=params)
        logger.success(
            f"{Fore.GREEN}{log_prefix} Leverage set/confirmed to {leverage}x (Category: {category}).{Style.RESET_ALL}"
        )
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        err_code = getattr(e, "code", None)
        # Bybit V5 code 110043: leverage not modified
        if err_code == 110043 or "leverage not modified" in err_str:
            logger.info(
                f"{Fore.CYAN}{log_prefix} Leverage for {symbol} is already set to {leverage}x.{Style.RESET_ALL}"
            )
            return True
        else:
            logger.error(
                f"{Fore.RED}{log_prefix} ExchangeError setting leverage: {e}{Style.RESET_ALL}"
            )
            return False  # Don't raise, return False
    except (ccxt.NetworkError, ccxt.AuthenticationError, ccxt.BadSymbol) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} API/Symbol error setting leverage: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, (ccxt.NetworkError)):
            raise e  # Allow retry for network errors
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error setting leverage: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] ERROR: Failed set leverage {leverage}x (Unexpected)",
            config.sms_config,
        )
        return False


# Snippet 3 / Function 3: Fetch USDT Balance (V5 UNIFIED)
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def fetch_usdt_balance(
    exchange: ccxt.bybit, config: Config
) -> tuple[Decimal, Decimal] | None:
    """Fetches the USDT balance (Total Equity and Available Balance) using Bybit V5 UNIFIED account logic."""
    func_name = "fetch_usdt_balance"
    log_prefix = f"[{func_name}]"
    usdt_symbol = config.api_config.usdt_symbol
    account_type_target = "UNIFIED"  # Hardcode for UTA

    logger.debug(
        f"{log_prefix} Fetching {account_type_target} account balance ({usdt_symbol})..."
    )
    try:
        # V5 requires specifying accountType
        balance_data = exchange.fetch_balance(
            params={"accountType": account_type_target}
        )
        # logger.debug(f"Raw balance response: {balance_data}") # Debugging

        # Parse V5 structure: info -> result -> list -> account dict -> coin list
        info = balance_data.get("info", {})
        result_list = info.get("result", {}).get("list", [])
        equity, available = None, None

        if not result_list:
            logger.warning(f"{log_prefix} Balance response list empty or missing.")
            return None

        unified_info = next(
            (
                acc
                for acc in result_list
                if acc.get("accountType") == account_type_target
            ),
            None,
        )
        if not unified_info:
            logger.warning(
                f"{log_prefix} Account type '{account_type_target}' not found in response."
            )
            return None

        equity = safe_decimal_conversion(
            unified_info.get("totalEquity"), context="Total Equity"
        )
        if equity is None:
            logger.warning(f"{log_prefix} Failed to parse total equity. Assuming 0.")

        coin_list = unified_info.get("coin", [])
        usdt_info = next((c for c in coin_list if c.get("coin") == usdt_symbol), None)
        if usdt_info:
            # Prioritize 'availableToWithdraw', fallback 'availableBalance'
            avail_str = usdt_info.get("availableToWithdraw") or usdt_info.get(
                "availableBalance"
            )
            available = safe_decimal_conversion(
                avail_str, context=f"{usdt_symbol} Available Balance"
            )
        else:
            logger.warning(
                f"{log_prefix} {usdt_symbol} details not found in UNIFIED coin list."
            )

        if available is None:
            logger.warning(
                f"{log_prefix} Failed to parse available {usdt_symbol} balance. Assuming 0."
            )

        final_equity = max(Decimal("0.0"), equity or Decimal("0.0"))
        final_available = max(Decimal("0.0"), available or Decimal("0.0"))

        logger.info(
            f"{Fore.GREEN}{log_prefix} OK - Equity: {final_equity:.4f}, Available {usdt_symbol}: {final_available:.4f}{Style.RESET_ALL}"
        )
        return final_equity, final_available

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Error fetching/parsing balance: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected error fetching balance: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            "[BybitHelper] CRITICAL: Failed fetch USDT balance!", config.sms_config
        )
        return None


# Snippet 4 / Function 4: Place Market Order with Slippage Check
@retry_api_call(
    max_retries_override=1, initial_delay_override=0
)  # Retry only once for market orders
def place_market_order_slippage_check(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    config: Config,
    max_slippage_pct_override: Decimal | None = None,
    is_reduce_only: bool = False,
    client_order_id: str | None = None,
) -> dict | None:
    """Places a market order on Bybit V5 after checking the current spread against a slippage threshold."""
    func_name = "place_market_order_slippage_check"
    market_base = symbol.split("/")[0]
    action = "CLOSE" if is_reduce_only else "ENTRY"
    log_prefix = f"[{func_name}({action} {side.upper()})]"
    api_conf = config.api_config
    effective_max_slippage = (
        max_slippage_pct_override
        if max_slippage_pct_override is not None
        else api_conf.default_slippage_pct
    )
    logger.info(
        f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}. Max Slippage: {effective_max_slippage:.4%}, ReduceOnly: {is_reduce_only}{Style.RESET_ALL}"
    )

    if amount <= api_conf.position_qty_epsilon:
        logger.error(
            f"{Fore.RED}{log_prefix}: Amount is zero or negative ({amount}). Aborting.{Style.RESET_ALL}"
        )
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category for {symbol}. Aborting.{Style.RESET_ALL}"
            )
            return None

        # --- Slippage Check ---
        if effective_max_slippage > 0:
            logger.debug(
                f"{log_prefix} Performing pre-order slippage check (Depth: {api_conf.shallow_ob_fetch_depth})..."
            )
            # Use the synchronous analyze_order_book utility
            ob_analysis = analyze_order_book(
                exchange,
                symbol,
                api_conf.shallow_ob_fetch_depth,
                api_conf.order_book_fetch_limit,
                config,
            )
            best_ask, best_bid = (
                ob_analysis.get("best_ask"),
                ob_analysis.get("best_bid"),
            )
            if best_bid and best_ask and best_bid > Decimal("0"):
                mid_price = (best_ask + best_bid) / 2
                spread_pct = (
                    ((best_ask - best_bid) / mid_price) * 100
                    if mid_price > 0
                    else Decimal("inf")
                )
                logger.debug(
                    f"{log_prefix} Current OB: Bid={format_price(exchange, symbol, best_bid)}, Ask={format_price(exchange, symbol, best_ask)}, Spread={spread_pct:.4%}"
                )
                if spread_pct > effective_max_slippage:
                    logger.error(
                        f"{Fore.RED}{log_prefix}: Aborted due to high spread {spread_pct:.4%} > Max {effective_max_slippage:.4%}.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}] ORDER ABORT ({side.upper()}): High Spread {spread_pct:.4%}",
                        config.sms_config,
                    )
                    return None
            else:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix}: Could not get valid OB data for slippage check. Proceeding cautiously.{Style.RESET_ALL}"
                )
        else:
            logger.debug(f"{log_prefix} Slippage check skipped.")

        # --- Prepare and Place Order ---
        amount_str = format_amount(exchange, symbol, amount)
        if amount_str is None or amount_str == "Error":
            raise ValueError("Failed to format amount.")
        amount_float = float(amount_str)

        params: dict[str, Any] = {"category": category}
        if is_reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            max_coid_len = 36
            original_len = len(client_order_id)
            valid_coid = client_order_id[:max_coid_len]
            params["clientOrderId"] = valid_coid
            if len(valid_coid) < original_len:
                logger.warning(
                    f"{log_prefix} Client OID truncated: '{valid_coid}' (Orig len: {original_len})"
                )

        bg = Back.GREEN if side == api_conf.side_buy else Back.RED
        fg = Fore.BLACK
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}*** PLACING MARKET {side.upper()} {'REDUCE' if is_reduce_only else 'ENTRY'}: {amount_str} {symbol} (Params: {params}) ***{Style.RESET_ALL}"
        )

        order = exchange.create_market_order(symbol, side, amount_float, params=params)

        order_id = order.get("id")
        client_oid_resp = order.get("clientOrderId", params.get("clientOrderId", "N/A"))
        status = order.get("status", "?")
        filled_qty = safe_decimal_conversion(order.get("filled", "0.0"))
        avg_price = safe_decimal_conversion(order.get("average"))
        cost = safe_decimal_conversion(order.get("cost"))
        fee = order.get("fee", {})  # Fee info might be nested

        logger.success(
            f"{Fore.GREEN}{log_prefix}: Submitted OK. ID: {format_order_id(order_id)}, ClientOID: {client_oid_resp}, Status: {status}, Filled Qty: {format_amount(exchange, symbol, filled_qty)}, Avg Px: {format_price(exchange, symbol, avg_price)}, Cost: {cost:.4f}, Fee: {fee}{Style.RESET_ALL}"
        )
        return order

    except (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.ExchangeError,
        ccxt.NetworkError,
    ) as e:
        logger.error(
            f"{Fore.RED}{log_prefix}: API Error: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}] ORDER FAIL ({side.upper()} {action}): {type(e).__name__}",
            config.sms_config,
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected error placing market order: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{market_base}] ORDER FAIL ({side.upper()} {action}): Unexpected {type(e).__name__}.",
            config.sms_config,
        )
        return None


# Snippet 5 / Function 5: Cancel All Open Orders
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def cancel_all_orders(
    exchange: ccxt.bybit,
    symbol: str,
    config: Config,
    reason: str = "Cleanup",
    order_filter: Literal["Order", "StopOrder", "tpslOrder"] | None = None,
) -> bool:
    """Cancels all open orders matching a filter for a specific symbol on Bybit V5."""
    func_name = "cancel_all_orders"
    symbol.split("/")[0]
    log_prefix = (
        f"[{func_name}({symbol}, Filter:{order_filter or 'All'}, Reason:{reason})]"
    )
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting cancellation...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category. Aborting.{Style.RESET_ALL}"
            )
            return False

        # V5 cancelAllOrders uses category, symbol (optional), settleCoin (optional), orderFilter (optional)
        params = {"category": category}
        if order_filter:
            params["orderFilter"] = order_filter
        # if symbol: params['symbol'] = market['id'] # Can specify symbol too

        logger.debug(f"{log_prefix} Calling cancelAllOrders with params: {params}")
        response = exchange.cancel_all_orders(
            symbol, params=params
        )  # Pass symbol here too
        logger.debug(f"{log_prefix} Raw response: {response}")

        # --- Parse V5 Response ---
        # Response structure: { retCode: 0, retMsg: 'OK', result: { list: [ { orderId: '...', clientOrderId: '...' }, ... ], success: '1'/'0' }, ... }
        result_data = response.get("result", {})
        cancelled_list = result_data.get("list", [])
        success_flag = result_data.get("success")  # Might indicate overall success

        if response.get("retCode") == 0:
            if cancelled_list:
                logger.success(
                    f"{Fore.GREEN}{log_prefix} Cancelled {len(cancelled_list)} orders successfully.{Style.RESET_ALL}"
                )
                for item in cancelled_list:
                    logger.debug(
                        f"  - Cancelled ID: {format_order_id(item.get('orderId'))}, ClientOID: {item.get('clientOrderId')}"
                    )
                return True
            elif success_flag == "1":  # Success flag but empty list
                logger.info(
                    f"{Fore.CYAN}{log_prefix} No open orders found matching filter to cancel (Success flag received).{Style.RESET_ALL}"
                )
                return True
            else:  # retCode 0 but no list and no clear success flag
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} cancelAllOrders returned success code (0) but no order list. Assuming no matching orders.{Style.RESET_ALL}"
                )
                return True
        else:
            # Handle specific error codes if needed
            # e.g., Code 10001: Parameter error (might indicate bad filter)
            # e.g., Code 10004: No orders found (sometimes returned as error)
            ret_msg = response.get("retMsg", "Unknown Error")
            if "order not found" in ret_msg.lower() or response.get("retCode") == 10004:
                logger.info(
                    f"{Fore.CYAN}{log_prefix} No open orders found matching filter to cancel (Error code received).{Style.RESET_ALL}"
                )
                return True
            else:
                logger.error(
                    f"{Fore.RED}{log_prefix} Failed. Code: {response.get('retCode')}, Msg: {ret_msg}{Style.RESET_ALL}"
                )
                return False

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} API error during cancel all: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during cancel all: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


# --- Added cancel_order function ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def cancel_order(
    exchange: ccxt.bybit,
    symbol: str,
    order_id: str,
    config: Config,
    order_filter: Literal["Order", "StopOrder", "tpslOrder"] | None = None,
) -> bool:
    """Cancels a single specific order by ID."""
    func_name = "cancel_order"
    log_prefix = f"[{func_name}({symbol}, ID:{format_order_id(order_id)}, Filter:{order_filter})]"
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting cancellation...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category. Aborting.{Style.RESET_ALL}"
            )
            return False

        params = {"category": category}
        # Bybit V5 cancelOrder might require orderFilter for Stop/TP/SL orders
        if order_filter:
            params["orderFilter"] = order_filter

        logger.debug(
            f"{log_prefix} Calling exchange.cancel_order with ID={order_id}, Symbol={symbol}, Params={params}"
        )
        response = exchange.cancel_order(order_id, symbol, params=params)
        logger.debug(f"{log_prefix} Raw response: {response}")

        # Check response, CCXT might raise OrderNotFound on failure
        # If no exception, assume success (CCXT often normalizes this)
        logger.success(
            f"{Fore.GREEN}{log_prefix} Successfully cancelled order.{Style.RESET_ALL}"
        )
        return True

    except ccxt.OrderNotFound as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Order already gone or not found: {e}{Style.RESET_ALL}"
        )
        return True  # Treat as success
    except ccxt.InvalidOrder as e:  # e.g., trying to cancel a filled order
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Invalid order state for cancellation (likely filled/rejected): {e}{Style.RESET_ALL}"
        )
        return True  # Treat as success (already closed/gone)
    except ccxt.ExchangeError as e:
        # Check specific codes if needed
        # e.g., 110001: Order does not exist
        err_code = getattr(e, "code", None)
        if err_code == 110001:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Order not found (via ExchangeError 110001).{Style.RESET_ALL}"
            )
            return True  # Treat as success
        logger.error(
            f"{Fore.RED}{log_prefix} API error cancelling: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        return False  # Don't raise, return failure
    except ccxt.NetworkError as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Network error cancelling: {e}{Style.RESET_ALL}"
        )
        raise e  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error cancelling: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


# --- Added fetch_order function ---
@retry_api_call(max_retries_override=3, initial_delay_override=0.5)
def fetch_order(
    exchange: ccxt.bybit,
    symbol: str,
    order_id: str,
    config: Config,
    order_filter: Literal["Order", "StopOrder", "tpslOrder"] | None = None,
) -> dict | None:
    """Fetches details for a single specific order by ID."""
    func_name = "fetch_order"
    log_prefix = f"[{func_name}({symbol}, ID:{format_order_id(order_id)}, Filter:{order_filter})]"
    logger.debug(f"{log_prefix} Attempting fetch...")
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"
            )
            return None

        params = {"category": category}
        # Bybit V5 fetchOrder might need orderFilter for Stop/TP/SL orders
        if order_filter:
            params["orderFilter"] = order_filter

        logger.debug(
            f"{log_prefix} Calling exchange.fetch_order with ID={order_id}, Symbol={symbol}, Params={params}"
        )
        order_data = exchange.fetch_order(order_id, symbol, params=params)

        if order_data:
            logger.debug(
                f"{log_prefix} Order data fetched. Status: {order_data.get('status')}"
            )
            return order_data
        else:
            # CCXT fetch_order usually raises OrderNotFound, so this case is less likely
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} fetch_order returned no data (but no exception). Order likely not found.{Style.RESET_ALL}"
            )
            return None

    except ccxt.OrderNotFound as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Order not found: {e}{Style.RESET_ALL}"
        )
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        err_code = getattr(e, "code", None)
        # Treat specific error codes as OrderNotFound
        if err_code == 110001 or "order does not exist" in str(e).lower():
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Order not found (via ExchangeError).{Style.RESET_ALL}"
            )
            return None
        logger.error(
            f"{Fore.RED}{log_prefix} API error fetching: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- fetch_ohlcv_paginated (Synchronous version) ---
def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    config: Config,
    since: int | None = None,
    limit_per_req: int = 1000,  # Bybit V5 max limit is 1000
    max_total_candles: int | None = None,
) -> pd.DataFrame | list[list] | None:
    """Fetches historical OHLCV data for a symbol using pagination (synchronous)."""
    func_name = "fetch_ohlcv_paginated"
    log_prefix = f"[{func_name}({symbol}, {timeframe})]"

    if not exchange.has.get("fetchOHLCV"):
        logger.error(
            f"{Fore.RED}{log_prefix} Exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}"
        )
        return None

    try:
        exchange.parse_timeframe(timeframe) * 1000
        if limit_per_req > 1000:
            logger.warning(
                f"{log_prefix} Requested limit_per_req ({limit_per_req}) > max (1000). Clamping."
            )
            limit_per_req = 1000

        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Could not determine category. Assuming 'linear'.{Style.RESET_ALL}"
            )
            category = "linear"  # Default assumption

        params = {"category": category}

        since_str = (
            pd.to_datetime(since, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
            if since
            else "Recent"
        )
        limit_str = str(max_total_candles) if max_total_candles else "All"
        logger.info(
            f"{Fore.BLUE}{log_prefix} Fetching... Since: {since_str}, Target Limit: {limit_str}{Style.RESET_ALL}"
        )

        all_candles: list[list] = []
        current_since = since
        request_count = 0
        max_requests = float("inf")
        if max_total_candles:
            max_requests = math.ceil(max_total_candles / limit_per_req)

        retry_conf = config.api_config  # Get retry settings from config
        retry_delay = retry_conf.retry_delay_seconds
        max_retries = retry_conf.retry_count

        while request_count < max_requests:
            if max_total_candles and len(all_candles) >= max_total_candles:
                logger.info(
                    f"{log_prefix} Reached target limit ({max_total_candles}). Fetch complete."
                )
                break

            request_count += 1
            fetch_limit = limit_per_req
            if max_total_candles:
                remaining_needed = max_total_candles - len(all_candles)
                fetch_limit = min(limit_per_req, remaining_needed)
                if fetch_limit <= 0:
                    break  # Already have enough

            logger.debug(
                f"{log_prefix} Fetch Chunk #{request_count}: Since={current_since}, Limit={fetch_limit}, Params={params}"
            )

            candles_chunk: list[list] | None = None
            last_fetch_error: Exception | None = None

            # Internal retry loop for fetching this specific chunk
            for attempt in range(max_retries + 1):
                try:
                    candles_chunk = exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=current_since,
                        limit=fetch_limit,
                        params=params,
                    )
                    last_fetch_error = None
                    break  # Success
                except (
                    ccxt.NetworkError,
                    ccxt.RequestTimeout,
                    ccxt.ExchangeNotAvailable,
                    ccxt.RateLimitExceeded,
                    ccxt.DDoSProtection,
                ) as e:
                    last_fetch_error = e
                    if attempt >= max_retries:
                        break  # Max retries reached for this chunk
                    current_delay = (
                        retry_delay * (2**attempt) * random.uniform(0.8, 1.2)
                    )
                    logger.warning(
                        f"{Fore.YELLOW}{log_prefix} API Error chunk #{request_count} (Try {attempt + 1}/{max_retries + 1}): {e}. Retrying in {current_delay:.2f}s...{Style.RESET_ALL}"
                    )
                    time.sleep(current_delay)
                except ccxt.ExchangeError as e:
                    last_fetch_error = e
                    logger.error(
                        f"{Fore.RED}{log_prefix} ExchangeError chunk #{request_count}: {e}. Aborting chunk.{Style.RESET_ALL}"
                    )
                    break
                except Exception as e:
                    last_fetch_error = e
                    logger.error(
                        f"{log_prefix} Unexpected fetch chunk #{request_count} err: {e}",
                        exc_info=True,
                    )
                    break

            if last_fetch_error:
                logger.error(
                    f"{Fore.RED}{log_prefix} Failed to fetch chunk #{request_count} after {max_retries + 1} attempts. Last Error: {last_fetch_error}{Style.RESET_ALL}"
                )
                logger.warning(
                    f"{log_prefix} Returning potentially incomplete data ({len(all_candles)} candles) due to fetch failure."
                )
                break  # Stop pagination

            if not candles_chunk:
                logger.debug(
                    f"{log_prefix} No more candles returned (Chunk #{request_count})."
                )
                break

            # Filter duplicates if necessary
            if all_candles and candles_chunk[0][0] <= all_candles[-1][0]:
                logger.debug(
                    f"{log_prefix} Overlap detected chunk #{request_count}. Filtering."
                )
                first_new_ts = all_candles[-1][0] + 1
                candles_chunk = [c for c in candles_chunk if c[0] >= first_new_ts]
                if not candles_chunk:
                    logger.debug(f"{log_prefix} Entire chunk was overlap/duplicate.")
                    continue  # Skip to next fetch if needed

            num_fetched = len(candles_chunk)
            logger.debug(
                f"{log_prefix} Fetched {num_fetched} new candles (Chunk #{request_count}). Total: {len(all_candles) + num_fetched}"
            )
            all_candles.extend(candles_chunk)

            if num_fetched < fetch_limit:
                logger.debug(
                    f"{log_prefix} Received fewer candles than requested. End of data likely reached."
                )
                break

            # Update 'since' for the next request based on the timestamp of the *last* candle received
            current_since = (
                candles_chunk[-1][0] + 1
            )  # Request starting *after* the last received timestamp

            # Add a small delay based on rate limit
            time.sleep(
                max(
                    0.05,
                    1.0
                    / (
                        exchange.rateLimit
                        if exchange.rateLimit and exchange.rateLimit > 0
                        else 10
                    ),
                )
            )

        # Process final list
        return _process_ohlcv_list(
            all_candles, func_name, symbol, timeframe, max_total_candles
        )

    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Initial setup error for OHLCV fetch: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected critical error during OHLCV pagination setup: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- _process_ohlcv_list (Helper for fetch_ohlcv_paginated) ---
def _process_ohlcv_list(
    candle_list: list[list],
    parent_func_name: str,
    symbol: str,
    timeframe: str,
    max_candles: int | None = None,
) -> pd.DataFrame | list[list] | None:
    """Internal helper to convert OHLCV list to validated pandas DataFrame or return list."""
    func_name = f"{parent_func_name}._process_ohlcv_list"
    log_prefix = f"[{func_name}({symbol}, {timeframe})]"

    if not candle_list:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} No candles collected. Returning empty.{Style.RESET_ALL}"
        )
        return pd.DataFrame() if PANDAS_AVAILABLE else []

    logger.debug(f"{log_prefix} Processing {len(candle_list)} raw candles...")
    try:
        # Sort and remove duplicates first
        candle_list.sort(key=lambda x: x[0])
        unique_candles_dict = {c[0]: c for c in candle_list}
        unique_candles = list(unique_candles_dict.values())
        if len(unique_candles) < len(candle_list):
            logger.debug(
                f"{log_prefix} Removed {len(candle_list) - len(unique_candles)} duplicate timestamps."
            )

        # Trim to max_candles if specified
        if max_candles and len(unique_candles) > max_candles:
            logger.debug(f"{log_prefix} Trimming final list to {max_candles} candles.")
            unique_candles = unique_candles[-max_candles:]

        if not PANDAS_AVAILABLE:
            logger.info(
                f"{Fore.GREEN}{log_prefix} Processed {len(unique_candles)} unique candles (returning list).{Style.RESET_ALL}"
            )
            return unique_candles

        # Process into DataFrame
        df = pd.DataFrame(
            unique_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(
            df["timestamp"], unit="ms", utc=True, errors="coerce"
        )
        df.dropna(
            subset=["datetime"], inplace=True
        )  # Drop rows where timestamp conversion failed
        if df.empty:
            raise ValueError("All timestamp conversions failed or list was empty.")

        df.set_index("datetime", inplace=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle NaNs (optional: forward fill or drop)
        nan_counts = df[numeric_cols].isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Found {total_nans} NaNs in numeric columns. Forward filling... (Counts: {nan_counts[nan_counts > 0].to_dict()}){Style.RESET_ALL}"
            )
            df[numeric_cols] = df[numeric_cols].ffill()
            df.dropna(
                subset=numeric_cols, inplace=True
            )  # Drop any remaining NaNs at the start

        if df.empty:
            logger.error(
                f"{Fore.RED}{log_prefix} Processed DataFrame is empty after cleaning.{Style.RESET_ALL}"
            )

        logger.success(
            f"{Fore.GREEN}{log_prefix} Processed {len(df)} valid candles into DataFrame.{Style.RESET_ALL}"
        )
        return df

    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Error processing OHLCV list: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        # Fallback to returning the unique candle list if DataFrame processing fails
        return unique_candles if "unique_candles" in locals() else candle_list


# --- place_limit_order_tif ---
@retry_api_call(
    max_retries_override=1, initial_delay_override=0
)  # Typically don't retry limit orders unless network error
def place_limit_order_tif(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    price: Decimal,
    config: Config,
    time_in_force: Literal["GTC", "IOC", "FOK", "PostOnly"] = "GTC",  # Use Literal Type
    is_reduce_only: bool = False,
    is_post_only: bool = False,
    client_order_id: str | None = None,
) -> dict | None:
    """Places a limit order on Bybit V5 with options for Time-In-Force, Post-Only, and Reduce-Only."""
    func_name = "place_limit_order_tif"
    log_prefix = f"[{func_name}({side.upper()})]"
    api_conf = config.api_config
    logger.info(
        f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} @ {format_price(exchange, symbol, price)} (TIF:{time_in_force}, Reduce:{is_reduce_only}, Post:{is_post_only})...{Style.RESET_ALL}"
    )

    if amount <= api_conf.position_qty_epsilon or price <= Decimal("0"):
        logger.error(f"{Fore.RED}{log_prefix}: Invalid amount/price.{Style.RESET_ALL}")
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"
            )
            return None

        amount_str = format_amount(exchange, symbol, amount)
        price_str = format_price(exchange, symbol, price)
        if any(v is None or v == "Error" for v in [amount_str, price_str]):
            raise ValueError("Invalid amount/price formatting.")
        amount_float = float(amount_str)
        price_float = float(price_str)

        params: dict[str, Any] = {"category": category}
        # Handle TIF and PostOnly flags correctly
        if time_in_force == "PostOnly":
            params["postOnly"] = True
            params["timeInForce"] = "GTC"  # PostOnly is a flag, TIF is usually GTC
        elif time_in_force in ["GTC", "IOC", "FOK"]:
            params["timeInForce"] = time_in_force
            if is_post_only:
                params["postOnly"] = True  # Allow separate postOnly flag
        else:
            logger.warning(
                f"[{func_name}] Unsupported TIF '{time_in_force}'. Using GTC."
            )
            params["timeInForce"] = "GTC"

        if is_reduce_only:
            params["reduceOnly"] = True

        if client_order_id:
            max_coid_len = 36
            original_len = len(client_order_id)
            valid_coid = client_order_id[:max_coid_len]
            params["clientOrderId"] = valid_coid
            if len(valid_coid) < original_len:
                logger.warning(
                    f"[{func_name}] Client OID truncated: '{valid_coid}' (Orig len: {original_len})"
                )

        logger.info(
            f"{Fore.CYAN}{log_prefix}: Placing -> Amt:{amount_float}, Px:{price_float}, Params:{params}{Style.RESET_ALL}"
        )
        order = exchange.create_limit_order(
            symbol, side, amount_float, price_float, params=params
        )

        order_id = order.get("id")
        client_oid_resp = order.get("clientOrderId", params.get("clientOrderId", "N/A"))
        status = order.get("status", "?")
        effective_tif = order.get("timeInForce", params.get("timeInForce", "?"))
        is_post_only_resp = order.get("postOnly", params.get("postOnly", False))
        logger.success(
            f"{Fore.GREEN}{log_prefix}: Limit order placed. ID:{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, TIF:{effective_tif}, Post:{is_post_only_resp}{Style.RESET_ALL}"
        )
        return order

    except ccxt.OrderImmediatelyFillable as e:
        # This happens if PostOnly is True and the order would match immediately
        if params.get("postOnly"):
            logger.warning(
                f"{Fore.YELLOW}{log_prefix}: PostOnly order failed (would fill immediately): {e}{Style.RESET_ALL}"
            )
            return None  # Return None as the order was rejected by the exchange
        else:  # Should not happen if PostOnly is False
            logger.error(
                f"{Fore.RED}{log_prefix}: Unexpected OrderImmediatelyFillable without PostOnly: {e}{Style.RESET_ALL}"
            )
            return None
    except (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.ExchangeError,
        ccxt.NetworkError,
    ) as e:
        logger.error(
            f"{Fore.RED}{log_prefix}: API Error: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): {type(e).__name__}",
            config.sms_config,
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): Unexpected {type(e).__name__}.",
            config.sms_config,
        )
        return None


# --- get_current_position_bybit_v5 ---
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def get_current_position_bybit_v5(
    exchange: ccxt.bybit, symbol: str, config: Config
) -> dict[str, Any]:
    """Fetches the current position details for a symbol using Bybit V5's fetchPositions logic."""
    func_name = "get_current_position"
    log_prefix = f"[{func_name}({symbol}, V5)]"
    api_conf = config.api_config
    default_position: dict[str, Any] = {
        "symbol": symbol,
        "side": api_conf.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
        "liq_price": None,
        "mark_price": None,
        "pnl_unrealized": None,
        "leverage": None,
        "info": {},
    }
    logger.debug(f"{log_prefix} Fetching position...")
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"
            )
            return default_position
        if not exchange.has.get("fetchPositions"):
            logger.error(
                f"{Fore.RED}{log_prefix} fetchPositions not available.{Style.RESET_ALL}"
            )
            return default_position

        # V5 fetchPositions requires category and optionally symbol
        params = {"category": category, "symbol": market_id}
        logger.debug(f"{log_prefix} Calling fetch_positions with params: {params}")
        # Fetch specific symbol for efficiency
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        # logger.debug(f"{log_prefix} Raw positions response: {fetched_positions}")

        active_position_data: dict | None = None
        # V5 fetchPositions returns a list, find the one matching the symbol and relevant mode (One-Way = index 0)
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            pos_symbol = pos_info.get("symbol")
            pos_v5_side = pos_info.get("side", "None")
            pos_size_str = pos_info.get("size")
            pos_idx = int(pos_info.get("positionIdx", -1))
            # Match symbol and ensure it's the One-Way position (idx=0) or primary hedge pos if mode allows
            if (
                pos_symbol == market_id and pos_v5_side != "None" and pos_idx == 0
            ):  # Assuming One-Way mode target
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                if (
                    pos_size is not None
                    and abs(pos_size) > api_conf.position_qty_epsilon
                ):
                    active_position_data = pos
                    logger.debug(f"{log_prefix} Found active One-Way (idx 0) position.")
                    break

        if active_position_data:
            try:
                # Parse standardized CCXT fields first, fallback to info dict
                side_std = active_position_data.get("side")  # 'long' or 'short'
                contracts = safe_decimal_conversion(
                    active_position_data.get("contracts")
                )
                entry_price = safe_decimal_conversion(
                    active_position_data.get("entryPrice")
                )
                mark_price = safe_decimal_conversion(
                    active_position_data.get("markPrice")
                )
                liq_price = safe_decimal_conversion(
                    active_position_data.get("liquidationPrice")
                )
                pnl = safe_decimal_conversion(active_position_data.get("unrealizedPnl"))
                leverage = safe_decimal_conversion(active_position_data.get("leverage"))
                info = active_position_data.get("info", {})  # Raw API data

                # Map CCXT side to internal constants
                position_side = (
                    api_conf.pos_long
                    if side_std == "long"
                    else (
                        api_conf.pos_short if side_std == "short" else api_conf.pos_none
                    )
                )
                quantity = abs(contracts) if contracts is not None else Decimal("0.0")

                # Fallback parsing from info dict if standard fields are missing (less likely with recent CCXT)
                if quantity <= api_conf.position_qty_epsilon:
                    pos_size_info = safe_decimal_conversion(info.get("size"))
                    if pos_size_info is not None:
                        quantity = abs(pos_size_info)
                if entry_price is None:
                    entry_price = safe_decimal_conversion(info.get("avgPrice"))
                if mark_price is None:
                    mark_price = safe_decimal_conversion(info.get("markPrice"))
                if liq_price is None:
                    liq_price = safe_decimal_conversion(info.get("liqPrice"))
                if pnl is None:
                    pnl = safe_decimal_conversion(info.get("unrealisedPnl"))
                if leverage is None:
                    leverage = safe_decimal_conversion(info.get("leverage"))
                if position_side == api_conf.pos_none:
                    side_info = info.get("side")  # Bybit uses 'Buy'/'Sell' in info
                    position_side = (
                        api_conf.pos_long
                        if side_info == "Buy"
                        else (
                            api_conf.pos_short
                            if side_info == "Sell"
                            else api_conf.pos_none
                        )
                    )

                # Final check
                if (
                    position_side == api_conf.pos_none
                    or quantity <= api_conf.position_qty_epsilon
                ):
                    logger.info(
                        f"{log_prefix} Position found but size/side negligible after parsing."
                    )
                    return default_position

                log_color = (
                    Fore.GREEN if position_side == api_conf.pos_long else Fore.RED
                )
                logger.info(
                    f"{log_color}{log_prefix} ACTIVE {position_side} {symbol}: Qty={format_amount(exchange, symbol, quantity)}, Entry={format_price(exchange, symbol, entry_price)}, Mark={format_price(exchange, symbol, mark_price)}, Liq~{format_price(exchange, symbol, liq_price)}, uPNL={format_price(exchange, api_conf.usdt_symbol, pnl)}, Lev={leverage}x{Style.RESET_ALL}"
                )
                return {
                    "symbol": symbol,
                    "side": position_side,
                    "qty": quantity,
                    "entry_price": entry_price,
                    "liq_price": liq_price,
                    "mark_price": mark_price,
                    "pnl_unrealized": pnl,
                    "leverage": leverage,
                    "info": info,
                }

            except Exception as parse_err:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Error parsing active pos: {parse_err}. Data: {str(active_position_data)[:300]}{Style.RESET_ALL}"
                )
                return default_position
        else:
            logger.info(f"{log_prefix} No active One-Way position found for {symbol}.")
            return default_position

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API Error fetching pos: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return default_position  # Return default on non-network API errors
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching pos: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return default_position


# --- close_position_reduce_only ---
@retry_api_call(max_retries_override=2, initial_delay_override=1)
def close_position_reduce_only(
    exchange: ccxt.bybit,
    symbol: str,
    config: Config,
    position_to_close: dict[str, Any] | None = None,
    reason: str = "Signal Close",
) -> dict[str, Any] | None:
    """Closes the current position for the given symbol using a reduce-only market order."""
    func_name = "close_position_reduce_only"
    market_base = symbol.split("/")[0]
    log_prefix = f"[{func_name}({symbol}, Reason:{reason})]"
    api_conf = config.api_config
    logger.info(f"{Fore.YELLOW}{log_prefix} Initiating close...{Style.RESET_ALL}")

    # --- Get Position State ---
    live_position_data: dict[str, Any]
    if position_to_close:
        logger.debug(f"{log_prefix} Using provided position state.")
        live_position_data = position_to_close
    else:
        logger.debug(f"{log_prefix} Fetching current position state...")
        live_position_data = get_current_position_bybit_v5(
            exchange, symbol, config
        )  # Fetch fresh state

    live_side = live_position_data.get("side", api_conf.pos_none)
    live_qty = live_position_data.get("qty", Decimal("0.0"))

    if live_side == api_conf.pos_none or live_qty <= api_conf.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} No active position validated or qty is zero. Aborting close.{Style.RESET_ALL}"
        )
        return None  # Indicate no action needed / nothing to close

    # Determine the side needed for the closing order
    close_order_side: Literal["buy", "sell"] = (
        api_conf.side_sell if live_side == api_conf.pos_long else api_conf.side_buy
    )

    # --- Place Closing Order ---
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            raise ValueError("Cannot determine category for close order.")

        qty_str = format_amount(exchange, symbol, live_qty)
        if qty_str is None or qty_str == "Error":
            raise ValueError("Failed formatting position quantity.")
        float(qty_str)

        # Use place_market_order helper for consistency and checks
        close_order = place_market_order_slippage_check(
            exchange=exchange,
            symbol=symbol,
            side=close_order_side,
            amount=live_qty,  # Pass Decimal amount
            config=config,
            is_reduce_only=True,
            client_order_id=f"close_{market_base}_{int(time.time())}"[
                -36:
            ],  # Generate client ID
            reason=f"Close {live_side} ({reason})",  # Pass reason for logging inside helper
        )

        if close_order and close_order.get("id"):
            fill_price = safe_decimal_conversion(close_order.get("average"))
            fill_qty = safe_decimal_conversion(close_order.get("filled", "0.0"))
            order_id = format_order_id(close_order.get("id"))
            logger.success(
                f"{Fore.GREEN}{Style.BRIGHT}{log_prefix} Close Order ({reason}) Submitted OK for {symbol}. ID:{order_id}, Filled:{format_amount(exchange, symbol, fill_qty)}/{qty_str}, AvgFill:{format_price(exchange, symbol, fill_price)}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] Closed {live_side} {qty_str} @ ~{format_price(exchange, symbol, fill_price)} ({reason}). ID:{order_id}",
                config.sms_config,
            )
            return close_order
        else:
            # place_market_order already logs errors
            logger.error(
                f"{Fore.RED}{log_prefix} Failed to submit close order via helper.{Style.RESET_ALL}"
            )
            # Check if the helper might have returned None due to slippage check failure
            # If so, the position might still be open.
            return None

    except (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
    ) as e:  # Should be less likely if place_market_order used
        logger.error(
            f"{Fore.RED}{log_prefix} Close Order Error ({reason}): {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}] CLOSE FAIL ({live_side}): {type(e).__name__}",
            config.sms_config,
        )
        if isinstance(e, ccxt.NetworkError):
            raise e
        return None
    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        err_code = getattr(e, "code", None)
        # Bybit V5 codes indicating position already closed or reduce failed due to size
        # 110025: Position is closed
        # 110045: Order would not reduce position size
        # 30086: order quantity is greater than the remaining position size (UTA?)
        if err_code in [110025, 110045, 30086] or any(
            code in error_str
            for code in [
                "position is closed",
                "order would not reduce",
                "position size is zero",
                "qty is larger than position size",
            ]
        ):
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Close Order ({reason}): Exchange indicates already closed/zero or reduce fail: {e}. Assuming closed.{Style.RESET_ALL}"
            )
            return None  # Treat as if closed successfully
        else:
            logger.error(
                f"{Fore.RED}{log_prefix} Close Order ExchangeError ({reason}): {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] CLOSE FAIL ({live_side}): ExchangeError Code {err_code}",
                config.sms_config,
            )
            return None
    except (ccxt.NetworkError, ValueError) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Close Order Network/Setup Error ({reason}): {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Close Order Unexpected Error ({reason}): {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{market_base}] CLOSE FAIL ({live_side}): Unexpected Error",
            config.sms_config,
        )
        return None


# --- fetch_funding_rate ---
@retry_api_call()
def fetch_funding_rate(
    exchange: ccxt.bybit, symbol: str, config: Config
) -> dict[str, Any] | None:
    """Fetches the current funding rate details for a perpetual swap symbol on Bybit V5."""
    func_name = "fetch_funding_rate"
    log_prefix = f"[{func_name}({symbol})]"
    logger.debug(f"{log_prefix} Fetching funding rate...")
    try:
        market = exchange.market(symbol)
        if not market.get("swap", False):
            logger.error(f"{Fore.RED}{log_prefix} Not a swap market.{Style.RESET_ALL}")
            return None
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Invalid category '{category}' for funding rate.{Style.RESET_ALL}"
            )
            return None

        params = {"category": category}
        logger.debug(f"{log_prefix} Calling fetch_funding_rate with params: {params}")
        funding_rate_info = exchange.fetch_funding_rate(
            symbol, params=params
        )  # Pass symbol here

        # Parse the standardized CCXT response
        processed_fr: dict[str, Any] = {
            "symbol": funding_rate_info.get("symbol"),
            "fundingRate": safe_decimal_conversion(
                funding_rate_info.get("fundingRate")
            ),
            "fundingTimestamp": funding_rate_info.get(
                "fundingTimestamp"
            ),  # ms timestamp of rate application
            "fundingDatetime": funding_rate_info.get(
                "fundingDatetime"
            ),  # ISO8601 string
            "markPrice": safe_decimal_conversion(funding_rate_info.get("markPrice")),
            "indexPrice": safe_decimal_conversion(funding_rate_info.get("indexPrice")),
            "nextFundingTime": funding_rate_info.get(
                "nextFundingTimestamp"
            ),  # ms timestamp of next funding
            "nextFundingDatetime": None,  # Will be populated below
            "info": funding_rate_info.get("info", {}),  # Raw exchange response
        }

        if processed_fr["fundingRate"] is None:
            logger.warning(f"{log_prefix} Could not parse 'fundingRate'.")
        if processed_fr["nextFundingTime"]:
            try:
                processed_fr["nextFundingDatetime"] = (
                    pd.to_datetime(
                        processed_fr["nextFundingTime"], unit="ms", utc=True
                    ).strftime("%Y-%m-%d %H:%M:%S %Z")
                    if PANDAS_AVAILABLE
                    else str(processed_fr["nextFundingTime"])
                )
            except Exception as dt_err:
                logger.warning(
                    f"{log_prefix} Could not format next funding datetime: {dt_err}"
                )

        rate = processed_fr.get("fundingRate")
        next_dt_str = processed_fr.get("nextFundingDatetime", "N/A")
        rate_str = f"{rate:.6%}" if rate is not None else "N/A"
        logger.info(f"{log_prefix} Funding Rate: {rate_str}. Next: {next_dt_str}")
        return processed_fr

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API Error fetching funding rate: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching funding rate: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- set_position_mode_bybit_v5 ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def set_position_mode_bybit_v5(
    exchange: ccxt.bybit,
    symbol_or_category: str,
    mode: Literal["one-way", "hedge"],
    config: Config,
) -> bool:
    """Sets the position mode (One-Way or Hedge) for a specific category (Linear/Inverse) on Bybit V5."""
    func_name = "set_position_mode"
    log_prefix = f"[{func_name}(Target:{mode})]"
    logger.info(
        f"{Fore.CYAN}{log_prefix} Setting mode '{mode}' for category of '{symbol_or_category}'...{Style.RESET_ALL}"
    )

    # Map mode string to Bybit API code (0 for One-Way, 3 for Hedge)
    mode_map = {"one-way": "0", "hedge": "3"}
    target_mode_code = mode_map.get(mode.lower())
    if target_mode_code is None:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid mode '{mode}'. Use 'one-way' or 'hedge'.{Style.RESET_ALL}"
        )
        return False

    # Determine target category
    target_category: Literal["linear", "inverse"] | None = None
    if symbol_or_category.lower() in ["linear", "inverse"]:
        target_category = symbol_or_category.lower()  # type: ignore
    else:
        try:
            market = exchange.market(symbol_or_category)
            category = _get_v5_category(market)
            if category in ["linear", "inverse"]:
                target_category = category  # type: ignore
        except Exception as e:
            logger.warning(
                f"{log_prefix} Could not get market/category for '{symbol_or_category}': {e}"
            )

    if not target_category:
        logger.error(
            f"{Fore.RED}{log_prefix} Could not determine contract category (linear/inverse) from '{symbol_or_category}'.{Style.RESET_ALL}"
        )
        return False

    logger.debug(
        f"{log_prefix} Target Category: {target_category}, Mode Code: {target_mode_code} ('{mode}')"
    )

    # --- Call V5 Endpoint ---
    # Requires calling a private endpoint not directly exposed by standard CCXT methods easily.
    # Use `exchange.private_post_v5_position_switch_mode`
    endpoint = "private_post_v5_position_switch_mode"
    if not hasattr(exchange, endpoint):
        logger.error(
            f"{Fore.RED}{log_prefix} CCXT version lacks '{endpoint}'. Cannot set mode via V5 API.{Style.RESET_ALL}"
        )
        return False

    params = {"category": target_category, "mode": target_mode_code}
    logger.debug(f"{log_prefix} Calling {endpoint} with params: {params}")
    try:
        response = getattr(exchange, endpoint)(params)
        logger.debug(f"{log_prefix} Raw V5 endpoint response: {response}")
        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "").lower()

        if ret_code == 0:
            logger.success(
                f"{Fore.GREEN}{log_prefix} Mode successfully set to '{mode}' for {target_category}.{Style.RESET_ALL}"
            )
            return True
        # Code 110021: Already in the target mode (or mode not modified)
        # Code 34036: Already in the target mode (specific to UTA?)
        elif ret_code in [110021, 34036] or "not modified" in ret_msg:
            logger.info(
                f"{Fore.CYAN}{log_prefix} Mode already set to '{mode}' for {target_category}.{Style.RESET_ALL}"
            )
            return True
        # Code 110020: Cannot switch mode with active positions/orders
        elif (
            ret_code == 110020
            or "have position" in ret_msg
            or "active order" in ret_msg
        ):
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot switch mode: Active position or orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"
            )
            return False
        else:
            # Raise unexpected error codes
            raise ccxt.ExchangeError(
                f"Bybit API error setting mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e:
        # Handle specific errors like active positions/orders without failing loudly
        if isinstance(e, ccxt.ExchangeError) and "110020" in str(e):
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot switch mode (active position/orders): {e}{Style.RESET_ALL}"
            )
            return False  # Return False clearly
        else:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} API Error setting mode: {e}{Style.RESET_ALL}"
            )
            if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)):
                raise e  # Retry network/auth errors
            return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error setting mode: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


# --- fetch_l2_order_book_validated ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def fetch_l2_order_book_validated(
    exchange: ccxt.bybit, symbol: str, limit: int, config: Config
) -> dict[str, list[tuple[Decimal, Decimal]]] | None:
    """Fetches the Level 2 order book for a symbol using Bybit V5 fetchOrderBook and validates the data."""
    func_name = "fetch_l2_order_book"
    log_prefix = f"[{func_name}({symbol}, Limit:{limit})]"
    logger.debug(f"{log_prefix} Fetching L2 OB...")

    if not exchange.has.get("fetchOrderBook"):
        logger.error(
            f"{Fore.RED}{log_prefix} fetchOrderBook not supported.{Style.RESET_ALL}"
        )
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"
            )
            return None
        params = {"category": category}

        # Check and clamp limit according to Bybit V5 category limits
        max_limit_map = {
            "spot": 200,
            "linear": 500,
            "inverse": 500,
            "option": 25,
        }  # Check Bybit docs for current limits
        max_limit = max_limit_map.get(category, 50)  # Default fallback
        if limit > max_limit:
            logger.warning(
                f"{log_prefix} Clamping limit {limit} to {max_limit} for category '{category}'."
            )
            limit = max_limit

        logger.debug(
            f"{log_prefix} Calling fetchOrderBook with limit={limit}, params={params}"
        )
        order_book = exchange.fetch_order_book(symbol, limit=limit, params=params)

        if (
            not isinstance(order_book, dict)
            or "bids" not in order_book
            or "asks" not in order_book
        ):
            raise ValueError("Invalid OB structure")
        raw_bids = order_book["bids"]
        raw_asks = order_book["asks"]
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
            raise ValueError("Bids/Asks not lists")

        # Validate and convert entries to Decimal tuples [price, amount]
        validated_bids: list[tuple[Decimal, Decimal]] = []
        validated_asks: list[tuple[Decimal, Decimal]] = []
        conversion_errors = 0
        for p_str, a_str in raw_bids:
            p = safe_decimal_conversion(p_str)
            a = safe_decimal_conversion(a_str)
            if not (p and a and p > 0 and a >= 0):
                conversion_errors += 1
                continue
            validated_bids.append((p, a))
        for p_str, a_str in raw_asks:
            p = safe_decimal_conversion(p_str)
            a = safe_decimal_conversion(a_str)
            if not (p and a and p > 0 and a >= 0):
                conversion_errors += 1
                continue
            validated_asks.append((p, a))

        if conversion_errors > 0:
            logger.warning(
                f"{log_prefix} Skipped {conversion_errors} invalid OB entries."
            )
        if not validated_bids or not validated_asks:
            logger.warning(
                f"{log_prefix} Empty validated bids/asks."
            )  # Return potentially empty lists

        # Check for crossed book
        if (
            validated_bids
            and validated_asks
            and validated_bids[0][0] >= validated_asks[0][0]
        ):
            logger.error(
                f"{Fore.RED}{log_prefix} OB crossed: Bid ({validated_bids[0][0]}) >= Ask ({validated_asks[0][0]}).{Style.RESET_ALL}"
            )
            # Return the crossed book for upstream handling

        logger.debug(
            f"{log_prefix} Processed L2 OB OK. Bids:{len(validated_bids)}, Asks:{len(validated_asks)}"
        )
        # Return validated data in a structure consistent with analysis needs
        return {
            "symbol": symbol,
            "bids": validated_bids,
            "asks": validated_asks,
            "timestamp": order_book.get("timestamp"),
            "datetime": order_book.get("datetime"),
            "nonce": order_book.get("nonce"),
        }

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API/Validation Error fetching L2 OB: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching L2 OB: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- place_native_stop_loss ---
@retry_api_call(max_retries_override=1, initial_delay_override=0)
def place_native_stop_loss(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    stop_price: Decimal,
    config: Config,
    trigger_by: Literal["LastPrice", "MarkPrice", "IndexPrice"] = "MarkPrice",
    client_order_id: str | None = None,
    position_idx: Literal[0, 1, 2] = 0,
) -> dict | None:
    """Places a native Stop Market order on Bybit V5, intended as a Stop Loss (reduceOnly)."""
    func_name = "place_native_stop_loss"
    log_prefix = f"[{func_name}({side.upper()})]"
    api_conf = config.api_config
    logger.info(
        f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trigger @ {format_price(exchange, symbol, stop_price)} ({trigger_by}), PosIdx:{position_idx}...{Style.RESET_ALL}"
    )

    if amount <= api_conf.position_qty_epsilon or stop_price <= Decimal("0"):
        logger.error(
            f"{Fore.RED}{log_prefix}: Invalid amount/stop price.{Style.RESET_ALL}"
        )
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"
            )
            return None

        amount_str = format_amount(exchange, symbol, amount)
        stop_price_str = format_price(exchange, symbol, stop_price)
        if any(v is None or v == "Error" for v in [amount_str, stop_price_str]):
            raise ValueError("Invalid amount/price formatting.")
        amount_float = float(amount_str)

        # --- V5 Stop Order Parameters ---
        # Use create_order with type='market' and stop loss params
        params: dict[str, Any] = {
            "category": category,
            "stopLoss": stop_price_str,  # Price level for the stop loss trigger
            "slTriggerBy": trigger_by,  # Trigger type (LastPrice, MarkPrice, IndexPrice)
            "reduceOnly": True,  # Ensure it only closes position
            "positionIdx": position_idx,  # Specify position index (0 for one-way)
            "tpslMode": "Full",  # Assume full position SL unless partial is needed
            "slOrderType": "Market",  # Execute as market order when triggered
            # 'slLimitPrice': '...' # Required if slOrderType='Limit'
        }
        if client_order_id:
            max_coid_len = 36
            valid_coid = client_order_id[:max_coid_len]
            params["orderLinkId"] = (
                valid_coid  # V5 uses orderLinkId for client ID on conditional orders
            )
            if len(valid_coid) < len(client_order_id):
                logger.warning(
                    f"{log_prefix} Client OID truncated to orderLinkId: '{valid_coid}'"
                )

        bg = Back.YELLOW
        fg = Fore.BLACK
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE Stop Loss (Market exec) -> Qty:{amount_float}, Side:{side}, TriggerPx:{stop_price_str}, TriggerBy:{trigger_by}, Params:{params}{Style.RESET_ALL}"
        )

        # create_order is used for placing conditional orders like stops in V5 via CCXT
        sl_order = exchange.create_order(
            symbol=symbol,
            type="market",  # Base order type is Market (triggered)
            side=side,  # The side of the order when triggered (e.g., sell for long SL)
            amount=amount_float,
            params=params,
        )

        order_id = sl_order.get("id")
        client_oid_resp = sl_order.get("info", {}).get(
            "orderLinkId", params.get("orderLinkId", "N/A")
        )
        status = sl_order.get("status", "?")
        returned_stop_price = safe_decimal_conversion(
            sl_order.get("stopPrice", info.get("stopLoss")), None
        )  # CCXT might use stopPrice
        returned_trigger = sl_order.get("trigger", trigger_by)  # CCXT might use trigger

        logger.success(
            f"{Fore.GREEN}{log_prefix}: Native SL order placed OK. ID:{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trigger:{format_price(exchange, symbol, returned_stop_price)} (by {returned_trigger}){Style.RESET_ALL}"
        )
        return sl_order

    except (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.BadSymbol,
    ) as e:
        logger.error(
            f"{Fore.RED}{log_prefix}: API Error placing SL: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected error placing SL: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] SL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}",
            config.sms_config,
        )
        return None


# --- fetch_open_orders_filtered ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def fetch_open_orders_filtered(
    exchange: ccxt.bybit,
    symbol: str,
    config: Config,
    side: Literal["buy", "sell"] | None = None,
    order_type: str | None = None,
    order_filter: Literal["Order", "StopOrder", "tpslOrder"] | None = None,
) -> list[dict] | None:
    """Fetches open orders for a specific symbol on Bybit V5, with optional filtering."""
    func_name = "fetch_open_orders_filtered"
    filter_log = f"(Side:{side or 'Any'}, Type:{order_type or 'Any'}, V5Filter:{order_filter or 'Default'})"
    log_prefix = f"[{func_name}({symbol}) {filter_log}]"
    logger.debug(f"{log_prefix} Fetching open orders...")
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"
            )
            return None

        params: dict[str, Any] = {"category": category}
        # V5 requires orderFilter to fetch conditional orders ('StopOrder' or 'tpslOrder')
        if order_filter:
            params["orderFilter"] = order_filter
        elif order_type:  # Infer filter from type if not provided
            norm_type = order_type.lower().replace("_", "").replace("-", "")
            if any(
                k in norm_type
                for k in ["stop", "trigger", "take", "tpsl", "conditional"]
            ):
                params["orderFilter"] = "StopOrder"
            else:
                params["orderFilter"] = "Order"  # Assume standard limit/market
        # else: fetch all types if no filter specified? Bybit default might be 'Order'. Let CCXT handle default.

        logger.debug(
            f"{log_prefix} Calling fetch_open_orders with symbol={symbol}, params={params}"
        )
        # Pass symbol to fetch for that specific market
        open_orders = exchange.fetch_open_orders(symbol=symbol, params=params)

        if not open_orders:
            logger.debug(f"{log_prefix} No open orders found matching criteria.")
            return []

        # --- Client-Side Filtering (if needed beyond API filter) ---
        filtered = open_orders
        initial_count = len(filtered)
        if side:
            side_lower = side.lower()
            filtered = [o for o in filtered if o.get("side", "").lower() == side_lower]
            logger.debug(
                f"{log_prefix} Filtered by side='{side}'. Count: {initial_count} -> {len(filtered)}."
            )
            initial_count = len(filtered)  # Update count for next filter log

        if order_type:
            norm_type_filter = order_type.lower().replace("_", "").replace("-", "")
            # Check standard 'type' and potentially 'info.orderType' or conditional types

            def check_type(o):
                o_type = o.get("type", "").lower().replace("_", "").replace("-", "")
                info = o.get("info", {})
                # Check standard type, info type, and conditional type fields
                return (
                    o_type == norm_type_filter
                    or info.get("orderType", "").lower() == norm_type_filter
                    or info.get("stopOrderType", "").lower() == norm_type_filter
                )

            filtered = [o for o in filtered if check_type(o)]
            logger.debug(
                f"{log_prefix} Filtered by type='{order_type}'. Count: {initial_count} -> {len(filtered)}."
            )

        logger.info(f"{log_prefix} Fetched/filtered {len(filtered)} open orders.")
        return filtered

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API Error fetching open orders: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching open orders: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- calculate_margin_requirement ---
def calculate_margin_requirement(
    exchange: ccxt.bybit,
    symbol: str,
    amount: Decimal,
    price: Decimal,
    leverage: Decimal,
    config: Config,
    order_side: Literal["buy", "sell"],
    is_maker: bool = False,
) -> tuple[Decimal | None, Decimal | None]:
    """Calculates the estimated Initial Margin (IM) requirement for placing an order on Bybit V5."""
    func_name = "calculate_margin_requirement"
    log_prefix = f"[{func_name}]"
    api_conf = config.api_config
    logger.debug(
        f"{log_prefix} Calc margin: {order_side} {format_amount(exchange, symbol, amount)} @ {format_price(exchange, symbol, price)}, Lev:{leverage}x, Maker:{is_maker}"
    )

    if amount <= 0 or price <= 0 or leverage <= 0:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid inputs (amount/price/leverage must be > 0).{Style.RESET_ALL}"
        )
        return None, None

    try:
        market = exchange.market(symbol)
        quote_currency = market.get("quote", api_conf.usdt_symbol)
        if not market.get("contract"):
            logger.error(
                f"{Fore.RED}{log_prefix} Not a contract symbol: {symbol}. Cannot calculate margin.{Style.RESET_ALL}"
            )
            return None, None

        # Calculate Order Value
        # Handle inverse contracts where value might be Base / Price
        is_inverse = market.get("inverse", False)
        position_value: Decimal
        if is_inverse:
            # Value = Amount (in Base currency contracts) / Price
            if price <= 0:
                raise ValueError(
                    "Price must be positive for inverse value calculation."
                )
            position_value = amount / price  # Result is in Quote currency terms
        else:
            # Value = Amount (in Base) * Price
            position_value = amount * price
        logger.debug(
            f"{log_prefix} Est Order Value: {format_price(exchange, quote_currency, position_value)} {quote_currency}"
        )

        # Initial Margin = Order Value / Leverage
        initial_margin_base = position_value / leverage
        logger.debug(
            f"{log_prefix} Base IM (Value/Lev): {format_price(exchange, quote_currency, initial_margin_base)} {quote_currency}"
        )

        # Estimate Fees (optional, adds buffer)
        fee_rate = api_conf.maker_fee_rate if is_maker else api_conf.taker_fee_rate
        estimated_fee = position_value * fee_rate
        logger.debug(
            f"{log_prefix} Est Fee ({fee_rate:.4%}): {format_price(exchange, quote_currency, estimated_fee)} {quote_currency}"
        )

        # Total Estimated IM = Base IM + Estimated Fee
        total_initial_margin_estimate = initial_margin_base  # + estimated_fee # Decide whether to include fee estimate

        logger.info(
            f"{log_prefix} Est TOTAL Initial Margin Req: {format_price(exchange, quote_currency, total_initial_margin_estimate)} {quote_currency}"
        )

        # Estimate Maintenance Margin (MM)
        maintenance_margin_estimate: Decimal | None = None
        try:
            # CCXT market structure often has maintenanceMarginRate under 'info' or directly
            mmr_keys = [
                "maintenanceMarginRate",
                "mmr",
                "maintMarginRatio",
            ]  # Common keys
            mmr_rate_str = None
            market_info = market.get("info", {})
            for key in mmr_keys:
                mmr_rate_str = market_info.get(key) or market.get(key)
                if mmr_rate_str is not None:
                    break

            if mmr_rate_str is not None:
                mmr_rate = safe_decimal_conversion(
                    mmr_rate_str, context=f"{symbol} MMR"
                )
                if mmr_rate is not None and mmr_rate >= 0:
                    maintenance_margin_estimate = position_value * mmr_rate
                    logger.debug(
                        f"{log_prefix} Basic MM Estimate (Base MMR {mmr_rate:.4%}): {format_price(exchange, quote_currency, maintenance_margin_estimate)} {quote_currency}"
                    )
                else:
                    logger.debug(
                        f"{log_prefix} Could not parse valid MMR rate from '{mmr_rate_str}'."
                    )
            else:
                logger.debug(f"{log_prefix} MMR key not found in market info.")
        except Exception as mm_err:
            logger.warning(f"{log_prefix} Could not estimate MM: {mm_err}")

        return total_initial_margin_estimate, maintenance_margin_estimate

    except (DivisionByZero, KeyError, ValueError) as e:
        logger.error(f"{Fore.RED}{log_prefix} Calculation error: {e}{Style.RESET_ALL}")
        return None, None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during margin calculation: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None, None


# --- fetch_ticker_validated (Fixed Timestamp/Age Logic) ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def fetch_ticker_validated(
    exchange: ccxt.bybit, symbol: str, config: Config, max_age_seconds: int = 30
) -> dict[str, Any] | None:
    """Fetches the ticker for a symbol from Bybit V5, validates its freshness and key values.
    Returns a dictionary with Decimal values, or None if validation fails or API error occurs.
    """
    func_name = "fetch_ticker_validated"
    log_prefix = f"[{func_name}({symbol})]"
    logger.debug(f"{log_prefix} Fetching/Validating ticker...")
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"
            )
            return None
        params = {"category": category}

        logger.debug(f"{log_prefix} Calling fetch_ticker with params: {params}")
        ticker = exchange.fetch_ticker(symbol, params=params)

        # --- Validation ---
        if not ticker:
            raise ValueError("fetch_ticker returned empty response.")

        timestamp_ms = ticker.get("timestamp")
        if timestamp_ms is None:
            raise ValueError(
                "Ticker data is missing timestamp."
            )  # Fail early if TS missing

        current_time_ms = time.time() * 1000
        age_seconds = (current_time_ms - timestamp_ms) / 1000.0

        # Check age validity
        if age_seconds > max_age_seconds:
            raise ValueError(
                f"Ticker data stale (Age: {age_seconds:.1f}s > Max: {max_age_seconds}s)."
            )
        if age_seconds < -10:  # Allow small future drift
            raise ValueError(
                f"Ticker timestamp ({timestamp_ms}) seems to be in the future (Age: {age_seconds:.1f}s)."
            )

        # Validate and convert key prices to Decimal
        last_price = safe_decimal_conversion(ticker.get("last"))
        bid_price = safe_decimal_conversion(ticker.get("bid"))
        ask_price = safe_decimal_conversion(ticker.get("ask"))
        if last_price is None or last_price <= 0:
            raise ValueError(f"Invalid 'last' price: {ticker.get('last')}")
        if bid_price is None or bid_price <= 0:
            logger.warning(f"{log_prefix} Invalid/missing 'bid': {ticker.get('bid')}")
        if ask_price is None or ask_price <= 0:
            logger.warning(f"{log_prefix} Invalid/missing 'ask': {ticker.get('ask')}")

        spread, spread_pct = None, None
        if bid_price and ask_price:
            if bid_price >= ask_price:
                logger.warning(
                    f"{log_prefix} Bid ({bid_price}) >= Ask ({ask_price}). Using NaN for spread."
                )  # Warn but don't fail
            else:
                spread = ask_price - bid_price
                mid_price = (ask_price + bid_price) / 2
                spread_pct = (
                    (spread / mid_price) * 100 if mid_price > 0 else Decimal("inf")
                )
        else:
            logger.warning(
                f"{log_prefix} Cannot calculate spread due to missing bid/ask."
            )

        # Convert other fields to Decimal safely
        validated_ticker = {
            "symbol": ticker.get("symbol", symbol),
            "timestamp": timestamp_ms,
            "datetime": ticker.get("datetime"),
            "last": last_price,
            "bid": bid_price,
            "ask": ask_price,
            "bidVolume": safe_decimal_conversion(ticker.get("bidVolume")),
            "askVolume": safe_decimal_conversion(ticker.get("askVolume")),
            "baseVolume": safe_decimal_conversion(ticker.get("baseVolume")),
            "quoteVolume": safe_decimal_conversion(ticker.get("quoteVolume")),
            "high": safe_decimal_conversion(ticker.get("high")),
            "low": safe_decimal_conversion(ticker.get("low")),
            "open": safe_decimal_conversion(ticker.get("open")),
            "close": last_price,  # Use last price as close for ticker
            "change": safe_decimal_conversion(ticker.get("change")),
            "percentage": safe_decimal_conversion(ticker.get("percentage")),
            "average": safe_decimal_conversion(ticker.get("average")),
            "vwap": safe_decimal_conversion(ticker.get("vwap")),
            "spread": spread,
            "spread_pct": spread_pct,
            "info": ticker.get("info", {}),
        }
        logger.debug(
            f"{log_prefix} Ticker OK: Last={format_price(exchange, symbol, last_price)}, Spread={(spread_pct or Decimal('NaN')):.4f}% (Age:{age_seconds:.1f}s)"
        )
        return validated_ticker

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API/Symbol error fetching ticker: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except ValueError as e:
        # Catch validation errors (stale, bad price, missing timestamp)
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Ticker validation failed: {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected ticker error: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- place_native_trailing_stop ---
@retry_api_call(max_retries_override=1, initial_delay_override=0)
def place_native_trailing_stop(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    trailing_offset: Decimal | str,
    config: Config,
    activation_price: Decimal | None = None,
    trigger_by: Literal["LastPrice", "MarkPrice", "IndexPrice"] = "MarkPrice",
    client_order_id: str | None = None,
    position_idx: Literal[0, 1, 2] = 0,
) -> dict | None:
    """Places a native Trailing Stop Market order on Bybit V5 (reduceOnly)."""
    func_name = "place_native_trailing_stop"
    log_prefix = f"[{func_name}({side.upper()})]"
    params: dict[str, Any] = {}
    trail_log_str = ""

    try:
        # --- Validate & Parse Trailing Offset ---
        if isinstance(trailing_offset, str) and trailing_offset.endswith("%"):
            percent_val = safe_decimal_conversion(trailing_offset.rstrip("%"))
            # Bybit V5 percentage trailing stop range (check docs, e.g., 0.1% to 10%)
            min_pct, max_pct = Decimal("0.1"), Decimal("10.0")
            if not (percent_val and min_pct <= percent_val <= max_pct):
                raise ValueError(
                    f"Percentage trail '{trailing_offset}' out of range ({min_pct}%-{max_pct}%)."
                )
            params["trailingStop"] = str(
                percent_val.quantize(Decimal("0.01"))
            )  # Format to 2 decimal places for %
            trail_log_str = f"{percent_val}%"
        elif isinstance(trailing_offset, Decimal):
            if trailing_offset <= Decimal("0"):
                raise ValueError(
                    f"Absolute trail delta must be positive: {trailing_offset}"
                )
            delta_str = format_price(
                exchange, symbol, trailing_offset
            )  # Use price precision for delta
            if delta_str is None or delta_str == "Error":
                raise ValueError("Invalid absolute trail delta formatting.")
            params["trailingStop"] = (
                delta_str  # V5 uses trailingStop for both % and absolute value
            )
            trail_log_str = f"{delta_str} (abs)"
        else:
            raise TypeError(f"Invalid trailing_offset type: {type(trailing_offset)}")

        if activation_price is not None and activation_price <= Decimal("0"):
            raise ValueError("Activation price must be positive.")

        logger.info(
            f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trail:{trail_log_str}, ActPx:{format_price(exchange, symbol, activation_price) or 'Immediate'}, Trigger:{trigger_by}, PosIdx:{position_idx}{Style.RESET_ALL}"
        )

        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"
            )
            return None

        amount_str = format_amount(exchange, symbol, amount)
        if amount_str is None or amount_str == "Error":
            raise ValueError("Invalid amount formatting.")
        amount_float = float(amount_str)

        activation_price_str = (
            format_price(exchange, symbol, activation_price)
            if activation_price is not None
            else None
        )
        if activation_price is not None and (
            activation_price_str is None or activation_price_str == "Error"
        ):
            raise ValueError("Invalid activation price formatting.")

        # --- V5 Trailing Stop Parameters ---
        # Use create_order with type='market' and trailing stop params
        params.update(
            {
                "category": category,
                "reduceOnly": True,
                "positionIdx": position_idx,
                "tpslMode": "Full",  # Assume full position TSL
                "triggerBy": trigger_by,
                # 'tsOrderType': 'Market' # This seems redundant if base type is Market
                # Trailing stop value/percentage already in params['trailingStop']
            }
        )
        if activation_price_str is not None:
            params["activePrice"] = activation_price_str
        if client_order_id:
            max_coid_len = 36
            valid_coid = client_order_id[:max_coid_len]
            params["orderLinkId"] = valid_coid  # Use orderLinkId for conditional orders
            if len(valid_coid) < len(client_order_id):
                logger.warning(
                    f"{log_prefix} Client OID truncated to orderLinkId: '{valid_coid}'"
                )

        bg = Back.YELLOW
        fg = Fore.BLACK
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE TSL (Market exec) -> Qty:{amount_float}, Side:{side}, Trail:{trail_log_str}, ActPx:{activation_price_str or 'Immediate'}, Params:{params}{Style.RESET_ALL}"
        )

        tsl_order = exchange.create_order(
            symbol=symbol,
            type="market",  # Base order type is Market (triggered)
            side=side,
            amount=amount_float,
            params=params,
        )

        order_id = tsl_order.get("id")
        client_oid_resp = tsl_order.get("info", {}).get(
            "orderLinkId", params.get("orderLinkId", "N/A")
        )
        status = tsl_order.get("status", "?")
        returned_trail = tsl_order.get("info", {}).get("trailingStop")
        returned_act = safe_decimal_conversion(
            tsl_order.get("info", {}).get("activePrice")
        )
        returned_trigger = tsl_order.get("trigger", trigger_by)

        logger.success(
            f"{Fore.GREEN}{log_prefix}: Native TSL order placed OK. ID:{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trail:{returned_trail}, ActPx:{format_price(exchange, symbol, returned_act)}, TriggerBy:{returned_trigger}{Style.RESET_ALL}"
        )
        return tsl_order

    except (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.BadSymbol,
        ValueError,
        TypeError,
    ) as e:
        logger.error(
            f"{Fore.RED}{log_prefix}: API/Input Error placing TSL: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected error placing TSL: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] TSL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}",
            config.sms_config,
        )
        return None


# --- fetch_account_info_bybit_v5 ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def fetch_account_info_bybit_v5(
    exchange: ccxt.bybit, config: Config
) -> dict[str, Any] | None:
    """Fetches general account information from Bybit V5 API (`/v5/account/info`)."""
    func_name = "fetch_account_info"
    log_prefix = f"[{func_name}(V5)]"
    logger.debug(f"{log_prefix} Fetching Bybit V5 account info...")
    endpoint = "private_get_v5_account_info"
    try:
        if hasattr(exchange, endpoint):
            logger.debug(f"{log_prefix} Using {endpoint} endpoint.")
            account_info_raw = getattr(exchange, endpoint)()
            logger.debug(
                f"{log_prefix} Raw Account Info response: {str(account_info_raw)[:400]}..."
            )
            ret_code = account_info_raw.get("retCode")
            ret_msg = account_info_raw.get("retMsg")
            if ret_code == 0 and "result" in account_info_raw:
                result = account_info_raw["result"]
                # Parse relevant fields (check Bybit docs for current fields)
                parsed_info = {
                    "unifiedMarginStatus": result.get(
                        "unifiedMarginStatus"
                    ),  # 1: Regular account; 2: UTA Pro; 3: UTA classic; 4: Default margin account; 5: Not upgraded to UTA
                    "marginMode": result.get(
                        "marginMode"
                    ),  # 0: regular margin; 1: portfolio margin (PM)
                    "dcpStatus": result.get("dcpStatus"),  # Disconnect-protect status
                    "timeWindow": result.get("timeWindow"),
                    "smtCode": result.get("smtCode"),
                    "isMasterTrader": result.get("isMasterTrader"),
                    "updateTime": result.get("updateTime"),
                    "rawInfo": result,  # Include raw data
                }
                logger.info(
                    f"{log_prefix} Account Info: UTA Status={parsed_info.get('unifiedMarginStatus', 'N/A')}, MarginMode={parsed_info.get('marginMode', 'N/A')}, DCP Status={parsed_info.get('dcpStatus', 'N/A')}"
                )
                return parsed_info
            else:
                raise ccxt.ExchangeError(
                    f"Failed fetch/parse account info. Code={ret_code}, Msg='{ret_msg}'"
                )
        else:
            logger.warning(
                f"{log_prefix} CCXT lacks '{endpoint}'. Using fallback fetch_accounts() (less detail)."
            )
            accounts = exchange.fetch_accounts()  # Standard CCXT method
            if accounts:
                logger.info(
                    f"{log_prefix} Fallback fetch_accounts(): {str(accounts[0])[:200]}..."
                )
                return accounts[0]
            else:
                logger.error(
                    f"{Fore.RED}{log_prefix} Fallback fetch_accounts() returned no data.{Style.RESET_ALL}"
                )
                return None
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API Error fetching account info: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching account info: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- validate_market ---
def validate_market(
    exchange: ccxt.bybit, symbol: str, config: Config, check_active: bool = True
) -> dict | None:
    """Validates if a symbol exists on the exchange, is active, and optionally matches expectations from config."""
    func_name = "validate_market"
    log_prefix = f"[{func_name}({symbol})]"
    api_conf = config.api_config
    eff_expected_type = api_conf.expected_market_type
    eff_expected_logic = api_conf.expected_market_logic
    require_contract = (
        eff_expected_type != "spot"
    )  # Require contract unless explicitly spot

    logger.debug(
        f"{log_prefix} Validating... Checks: Type='{eff_expected_type or 'Any'}', Logic='{eff_expected_logic or 'Any'}', Active={check_active}, Contract={require_contract}"
    )
    try:
        # Load markets if not already loaded (should be done during init)
        if not exchange.markets:
            logger.info(f"{log_prefix} Markets not loaded. Loading...")
            exchange.load_markets(reload=True)
        if not exchange.markets:
            logger.error(
                f"{Fore.RED}{log_prefix} Failed to load markets.{Style.RESET_ALL}"
            )
            return None

        market = exchange.market(symbol)  # Throws BadSymbol if not found
        is_active = market.get("active", False)

        if check_active and not is_active:
            # Inactive markets might still be useful for historical data, treat as warning
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Validation Warning: Market is INACTIVE.{Style.RESET_ALL}"
            )
            # return None # Optionally fail validation for inactive markets

        actual_type = market.get("type")  # e.g., 'spot', 'swap'
        if eff_expected_type and actual_type != eff_expected_type:
            logger.error(
                f"{Fore.RED}{log_prefix} Validation Failed: Type mismatch. Expected '{eff_expected_type}', Got '{actual_type}'.{Style.RESET_ALL}"
            )
            return None

        is_contract = market.get("contract", False)
        if require_contract and not is_contract:
            logger.error(
                f"{Fore.RED}{log_prefix} Validation Failed: Expected a contract, but market type is '{actual_type}'.{Style.RESET_ALL}"
            )
            return None

        actual_logic_str: str | None = None
        if is_contract:
            actual_logic_str = _get_v5_category(market)  # linear/inverse
            if eff_expected_logic and actual_logic_str != eff_expected_logic:
                logger.error(
                    f"{Fore.RED}{log_prefix} Validation Failed: Logic mismatch. Expected '{eff_expected_logic}', Got '{actual_logic_str}'.{Style.RESET_ALL}"
                )
                return None

        logger.info(
            f"{Fore.GREEN}{log_prefix} Market OK: Type:{actual_type}, Logic:{actual_logic_str or 'N/A'}, Active:{is_active}.{Style.RESET_ALL}"
        )
        return market

    except ccxt.BadSymbol as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Validation Failed: Symbol not found. Error: {e}{Style.RESET_ALL}"
        )
        return None
    except ccxt.NetworkError as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Network error during market validation: {e}{Style.RESET_ALL}"
        )
        return None  # Network errors are usually critical for validation
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error validating market: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- fetch_recent_trades ---
@retry_api_call(max_retries_override=2, initial_delay_override=0.5)
def fetch_recent_trades(
    exchange: ccxt.bybit,
    symbol: str,
    config: Config,
    limit: int = 100,
    min_size_filter: Decimal | None = None,
) -> list[dict] | None:
    """Fetches recent public trades for a symbol from Bybit V5, validates data."""
    func_name = "fetch_recent_trades"
    log_prefix = f"[{func_name}({symbol}, limit={limit})]"
    api_conf = config.api_config
    filter_log = f"(MinSize:{format_amount(exchange, symbol, min_size_filter) if min_size_filter else 'N/A'})"
    logger.debug(f"{log_prefix} Fetching {limit} trades {filter_log}...")

    # Bybit V5 limit is 1000 for public trades
    max_limit = 1000
    if limit > max_limit:
        logger.warning(f"{log_prefix} Clamping limit {limit} to {max_limit}.")
        limit = max_limit
    if limit <= 0:
        logger.warning(f"{log_prefix} Invalid limit {limit}. Using 100.")
        limit = 100

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category.{Style.RESET_ALL}"
            )
            return None
        params = {"category": category}

        logger.debug(
            f"{log_prefix} Calling fetch_trades with limit={limit}, params={params}"
        )
        trades_raw = exchange.fetch_trades(symbol, limit=limit, params=params)

        if not trades_raw:
            logger.debug(f"{log_prefix} No recent trades found.")
            return []

        processed_trades: list[dict] = []
        conversion_errors = 0
        filtered_out_count = 0
        for trade in trades_raw:
            try:
                amount = safe_decimal_conversion(trade.get("amount"))
                price = safe_decimal_conversion(trade.get("price"))
                # Basic validation of core fields
                if (
                    not all(
                        [
                            trade.get("id"),
                            trade.get("timestamp"),
                            trade.get("side"),
                            price,
                            amount,
                        ]
                    )
                    or price <= 0
                    or amount <= 0
                ):
                    conversion_errors += 1
                    continue

                # Apply size filter
                if min_size_filter is not None and amount < min_size_filter:
                    filtered_out_count += 1
                    continue

                # Calculate cost if missing or seems incorrect
                cost = safe_decimal_conversion(trade.get("cost"))
                if (
                    cost is None
                    or abs(cost - (price * amount))
                    > api_conf.position_qty_epsilon * price
                ):
                    cost = price * amount  # Recalculate

                processed_trades.append(
                    {
                        "id": trade.get("id"),
                        "timestamp": trade.get("timestamp"),
                        "datetime": trade.get("datetime"),
                        "symbol": trade.get("symbol", symbol),
                        "side": trade.get("side"),
                        "price": price,
                        "amount": amount,
                        "cost": cost,
                        "takerOrMaker": trade.get("takerOrMaker"),
                        "fee": trade.get("fee"),  # Include fee if available
                        "info": trade.get("info", {}),
                    }
                )
            except Exception as proc_err:
                conversion_errors += 1
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Error processing single trade: {proc_err}. Data: {trade}{Style.RESET_ALL}"
                )

        if conversion_errors > 0:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Skipped {conversion_errors} trades due to processing errors.{Style.RESET_ALL}"
            )
        if filtered_out_count > 0:
            logger.debug(
                f"{log_prefix} Filtered {filtered_out_count} trades smaller than {min_size_filter}."
            )

        # Sort by timestamp descending (most recent first) - CCXT usually returns this way
        processed_trades.sort(key=lambda x: x["timestamp"], reverse=True)

        logger.info(
            f"{log_prefix} Fetched/processed {len(processed_trades)} trades {filter_log}."
        )
        return processed_trades

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API Error fetching trades: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching trades: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- update_limit_order ---
@retry_api_call(max_retries_override=1, initial_delay_override=0)
def update_limit_order(
    exchange: ccxt.bybit,
    symbol: str,
    order_id: str,
    config: Config,
    new_amount: Decimal | None = None,
    new_price: Decimal | None = None,
    new_trigger_price: Decimal | None = None,  # For conditional orders
    new_client_order_id: str | None = None,
) -> dict | None:
    """Attempts to modify an existing open limit or conditional order on Bybit V5."""
    func_name = "update_limit_order"
    log_prefix = f"[{func_name}(ID:{format_order_id(order_id)})]"
    api_conf = config.api_config

    # Check if anything is actually being changed
    if all(
        v is None
        for v in [new_amount, new_price, new_trigger_price, new_client_order_id]
    ):
        logger.warning(
            f"{log_prefix} No changes provided (amount, price, trigger, client ID). Aborting update."
        )
        return None

    # Basic validation of new values
    if new_amount is not None and new_amount <= api_conf.position_qty_epsilon:
        logger.error(f"{Fore.RED}{log_prefix}: Invalid new amount ({new_amount}).")
        return None
    if new_price is not None and new_price <= Decimal("0"):
        logger.error(f"{Fore.RED}{log_prefix}: Invalid new price ({new_price}).")
        return None
    if new_trigger_price is not None and new_trigger_price <= Decimal("0"):
        logger.error(
            f"{Fore.RED}{log_prefix}: Invalid new trigger price ({new_trigger_price})."
        )
        return None

    log_changes = []
    if new_amount is not None:
        log_changes.append(f"Amt:{format_amount(exchange, symbol, new_amount)}")
    if new_price is not None:
        log_changes.append(f"Px:{format_price(exchange, symbol, new_price)}")
    if new_trigger_price is not None:
        log_changes.append(
            f"TrigPx:{format_price(exchange, symbol, new_trigger_price)}"
        )
    if new_client_order_id is not None:
        log_changes.append("ClientOID")
    logger.info(
        f"{Fore.CYAN}{log_prefix}: Update {symbol} ({', '.join(log_changes)})...{Style.RESET_ALL}"
    )

    try:
        if not exchange.has.get("editOrder"):
            logger.error(
                f"{Fore.RED}{log_prefix}: editOrder not supported by this CCXT version/exchange config.{Style.RESET_ALL}"
            )
            return None

        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            raise ValueError(f"Cannot determine category for {symbol}")

        # --- Prepare Params for edit_order ---
        # Note: edit_order in CCXT might not directly support all V5 amendment features.
        # It often cancels and replaces for complex changes. Check CCXT Bybit implementation.
        # V5 amend endpoint: /v5/order/amend
        # Required: category, symbol, orderId OR orderLinkId
        # Optional: qty, price, triggerPrice, sl/tp settings, orderLinkId
        edit_params: dict[str, Any] = {"category": category}

        # Format new values if provided
        final_amount_str = (
            format_amount(exchange, symbol, new_amount)
            if new_amount is not None
            else None
        )
        final_price_str = (
            format_price(exchange, symbol, new_price) if new_price is not None else None
        )
        final_trigger_str = (
            format_price(exchange, symbol, new_trigger_price)
            if new_trigger_price is not None
            else None
        )

        # Add formatted values to params if they were provided
        if final_amount_str and final_amount_str != "Error":
            edit_params["qty"] = final_amount_str
        if final_price_str and final_price_str != "Error":
            edit_params["price"] = final_price_str
        if final_trigger_str and final_trigger_str != "Error":
            edit_params["triggerPrice"] = final_trigger_str

        if new_client_order_id:
            max_coid_len = 36
            valid_coid = new_client_order_id[:max_coid_len]
            edit_params["orderLinkId"] = valid_coid  # V5 amend uses orderLinkId
            if len(valid_coid) < len(new_client_order_id):
                logger.warning(
                    f"{log_prefix} New Client OID truncated to orderLinkId: '{valid_coid}'"
                )

        # Fetch current order to get side/type if needed by edit_order (CCXT specific)
        # current_order = fetch_order(exchange, symbol, order_id, config) # Use helper
        # if not current_order: raise ccxt.OrderNotFound(f"{log_prefix} Original order not found, cannot edit.")
        # status = current_order.get('status'); order_type = current_order.get('type')
        # if status != 'open': raise ccxt.InvalidOrder(f"{log_prefix}: Status is '{status}' (not 'open'). Cannot edit.")
        # --- edit_order might not need side/type if ID is sufficient ---

        logger.info(
            f"{Fore.CYAN}{log_prefix} Submitting update via edit_order. Params: {edit_params}{Style.RESET_ALL}"
        )

        # Use CCXT's edit_order method
        # Pass None for parameters not being changed (amount, price)
        # CCXT might require side/type from original order, check its specific implementation
        updated_order = exchange.edit_order(
            id=order_id,
            symbol=symbol,
            # type=current_order['type'], # May need original type
            # side=current_order['side'], # May need original side
            amount=float(final_amount_str)
            if final_amount_str
            else None,  # Pass float or None
            price=float(final_price_str)
            if final_price_str
            else None,  # Pass float or None
            params=edit_params,  # Pass category and trigger price etc. here
        )

        if updated_order:
            # edit_order might return the amended order OR the cancel/replace new order ID
            new_id = updated_order.get("id", order_id)
            status_after = updated_order.get("status", "?")
            new_client_oid_resp = updated_order.get("info", {}).get(
                "orderLinkId", edit_params.get("orderLinkId", "N/A")
            )
            logger.success(
                f"{Fore.GREEN}{log_prefix} Update OK. NewID:{format_order_id(new_id)}, Status:{status_after}, ClientOID:{new_client_oid_resp}{Style.RESET_ALL}"
            )
            return updated_order
        else:
            # Should not happen if edit_order raises exceptions on failure
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} edit_order returned no data. Check status manually.{Style.RESET_ALL}"
            )
            return None

    except (
        ccxt.OrderNotFound,
        ccxt.InvalidOrder,
        ccxt.NotSupported,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.BadSymbol,
        ValueError,
    ) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Failed update: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}{log_prefix} Unexpected update error: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- fetch_position_risk_bybit_v5 ---
@retry_api_call(max_retries_override=3, initial_delay_override=1.0)
def fetch_position_risk_bybit_v5(
    exchange: ccxt.bybit, symbol: str, config: Config
) -> dict[str, Any] | None:
    """Fetches detailed risk metrics for the current position of a specific symbol using Bybit V5 logic."""
    func_name = "fetch_position_risk"
    log_prefix = f"[{func_name}({symbol}, V5)]"
    api_conf = config.api_config
    logger.debug(f"{log_prefix} Fetching position risk...")
    default_risk = {
        "symbol": symbol,
        "side": api_conf.pos_none,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
        "mark_price": None,
        "liq_price": None,
        "leverage": None,
        "initial_margin": None,
        "maint_margin": None,
        "unrealized_pnl": None,
        "imr": None,
        "mmr": None,
        "position_value": None,
        "risk_limit_value": None,
        "info": {},
    }
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Not a contract symbol.{Style.RESET_ALL}"
            )
            return default_risk

        params = {"category": category, "symbol": market_id}
        position_data: list[dict] | None = None
        fetch_method_used = "N/A"

        # Prefer fetchPositionsRisk if available
        if exchange.has.get("fetchPositionsRisk"):
            try:
                logger.debug(f"{log_prefix} Using fetch_positions_risk...")
                position_data = exchange.fetch_positions_risk(
                    symbols=[symbol], params=params
                )
                fetch_method_used = "fetchPositionsRisk"
            except (ccxt.NotSupported, ccxt.ExchangeError) as e:
                logger.warning(
                    f"{log_prefix} fetch_positions_risk failed ({type(e).__name__}). Falling back."
                )
                position_data = None
        else:
            logger.debug(f"{log_prefix} fetchPositionsRisk not supported.")

        # Fallback to fetchPositions
        if position_data is None:
            if exchange.has.get("fetchPositions"):
                logger.debug(f"{log_prefix} Falling back to fetch_positions...")
                position_data = exchange.fetch_positions(
                    symbols=[symbol], params=params
                )
                fetch_method_used = "fetchPositions (Fallback)"
            else:
                logger.error(
                    f"{Fore.RED}{log_prefix} No position fetch methods available.{Style.RESET_ALL}"
                )
                return default_risk

        if position_data is None:
            logger.error(
                f"{Fore.RED}{log_prefix} Failed fetch position data ({fetch_method_used}).{Style.RESET_ALL}"
            )
            return default_risk

        # Find the active One-Way position (index 0)
        active_pos_risk: dict | None = None
        for pos in position_data:
            pos_info = pos.get("info", {})
            pos_symbol = pos_info.get("symbol")
            pos_v5_side = pos_info.get("side", "None")
            pos_size_str = pos_info.get("size")
            pos_idx = int(pos_info.get("positionIdx", -1))
            if pos_symbol == market_id and pos_v5_side != "None" and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                if (
                    pos_size is not None
                    and abs(pos_size) > api_conf.position_qty_epsilon
                ):
                    active_pos_risk = pos
                    logger.debug(
                        f"{log_prefix} Found active One-Way pos risk data ({fetch_method_used})."
                    )
                    break

        if not active_pos_risk:
            logger.info(f"{log_prefix} No active One-Way position found.")
            return default_risk

        # --- Parse Risk Data ---
        # Prioritize standardized CCXT fields, fallback to 'info'
        try:
            info = active_pos_risk.get("info", {})
            size = safe_decimal_conversion(
                active_pos_risk.get("contracts", info.get("size"))
            )
            entry_price = safe_decimal_conversion(
                active_pos_risk.get("entryPrice", info.get("avgPrice"))
            )
            mark_price = safe_decimal_conversion(
                active_pos_risk.get("markPrice", info.get("markPrice"))
            )
            liq_price = safe_decimal_conversion(
                active_pos_risk.get("liquidationPrice", info.get("liqPrice"))
            )
            leverage = safe_decimal_conversion(
                active_pos_risk.get("leverage", info.get("leverage"))
            )
            initial_margin = safe_decimal_conversion(
                active_pos_risk.get("initialMargin", info.get("positionIM"))
            )  # IM for the position
            maint_margin = safe_decimal_conversion(
                active_pos_risk.get("maintenanceMargin", info.get("positionMM"))
            )  # MM for the position
            pnl = safe_decimal_conversion(
                active_pos_risk.get("unrealizedPnl", info.get("unrealisedPnl"))
            )
            imr = safe_decimal_conversion(
                active_pos_risk.get("initialMarginPercentage", info.get("imr"))
            )  # Initial Margin Rate
            mmr = safe_decimal_conversion(
                active_pos_risk.get("maintenanceMarginPercentage", info.get("mmr"))
            )  # Maintenance Margin Rate
            pos_value = safe_decimal_conversion(
                active_pos_risk.get("contractsValue", info.get("positionValue"))
            )  # Value of the position
            risk_limit = safe_decimal_conversion(
                info.get("riskLimitValue")
            )  # Current risk limit tier value

            side_std = active_pos_risk.get("side")  # CCXT standard 'long'/'short'
            side_info = info.get("side")  # Bybit 'Buy'/'Sell'
            position_side = (
                api_conf.pos_long
                if side_std == "long" or side_info == "Buy"
                else (
                    api_conf.pos_short
                    if side_std == "short" or side_info == "Sell"
                    else api_conf.pos_none
                )
            )
            quantity = abs(size) if size is not None else Decimal("0.0")

            if (
                position_side == api_conf.pos_none
                or quantity <= api_conf.position_qty_epsilon
            ):
                logger.info(f"{log_prefix} Parsed pos {symbol} negligible.")
                return default_risk

            # --- Log Parsed Risk Info ---
            log_color = Fore.GREEN if position_side == api_conf.pos_long else Fore.RED
            quote_curr = market.get("quote", api_conf.usdt_symbol)
            logger.info(
                f"{log_color}{log_prefix} Position Risk ({position_side}):{Style.RESET_ALL}"
            )
            logger.info(
                f"  Qty:{format_amount(exchange, symbol, quantity)}, Entry:{format_price(exchange, symbol, entry_price)}, Mark:{format_price(exchange, symbol, mark_price)}"
            )
            logger.info(
                f"  Liq:{format_price(exchange, symbol, liq_price)}, Lev:{leverage}x, uPNL:{format_price(exchange, quote_curr, pnl)}"
            )
            logger.info(
                f"  IM:{format_price(exchange, quote_curr, initial_margin)}, MM:{format_price(exchange, quote_curr, maint_margin)}"
            )
            logger.info(
                f"  IMR:{imr:.4% if imr else 'N/A'}, MMR:{mmr:.4% if mmr else 'N/A'}, Value:{format_price(exchange, quote_curr, pos_value)}"
            )
            logger.info(f"  RiskLimitValue:{risk_limit or 'N/A'}")

            return {
                "symbol": symbol,
                "side": position_side,
                "qty": quantity,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "liq_price": liq_price,
                "leverage": leverage,
                "initial_margin": initial_margin,
                "maint_margin": maint_margin,
                "unrealized_pnl": pnl,
                "imr": imr,
                "mmr": mmr,
                "position_value": pos_value,
                "risk_limit_value": risk_limit,
                "info": info,
            }
        except Exception as parse_err:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Error parsing pos risk: {parse_err}. Data: {str(active_pos_risk)[:300]}{Style.RESET_ALL}"
            )
            return default_risk

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} API Error fetching pos risk: {e}{Style.RESET_ALL}"
        )
        if isinstance(e, ccxt.NetworkError):
            raise e  # Allow retry
        return default_risk
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching pos risk: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return default_risk


# --- set_isolated_margin_bybit_v5 ---
@retry_api_call(max_retries_override=2, initial_delay_override=1.0)
def set_isolated_margin_bybit_v5(
    exchange: ccxt.bybit, symbol: str, leverage: int, config: Config
) -> bool:
    """Sets margin mode to ISOLATED for a specific symbol on Bybit V5 and sets leverage for it."""
    func_name = "set_isolated_margin"
    log_prefix = f"[{func_name}({symbol}, {leverage}x)]"
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting ISOLATED mode...{Style.RESET_ALL}")
    ret_code = -1  # For tracking API response code
    if leverage <= 0:
        logger.error(
            f"{Fore.RED}{log_prefix} Leverage must be positive.{Style.RESET_ALL}"
        )
        return False
    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}{log_prefix} Not a contract symbol ({category}).{Style.RESET_ALL}"
            )
            return False

        # --- Attempt via unified set_margin_mode first ---
        try:
            logger.debug(
                f"{log_prefix} Attempting via unified exchange.set_margin_mode..."
            )
            # Pass category and leverage directly if supported by CCXT method
            exchange.set_margin_mode(
                marginMode="isolated",
                symbol=symbol,
                params={"category": category, "leverage": leverage},
            )
            logger.success(
                f"{Fore.GREEN}{log_prefix} Isolated mode & leverage {leverage}x set OK via unified call for {symbol}.{Style.RESET_ALL}"
            )
            return True  # Assume success if no exception
        except (
            ccxt.NotSupported,
            ccxt.ExchangeError,
            ccxt.ArgumentsRequired,
        ) as e_unified:
            # Log failure and proceed to V5 specific endpoint attempt
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Unified set_margin_mode failed: {e_unified}. Trying private V5 endpoint...{Style.RESET_ALL}"
            )

        # --- Fallback to private V5 endpoint ---
        endpoint = "private_post_v5_position_switch_isolated"
        if not hasattr(exchange, endpoint):
            logger.error(
                f"{Fore.RED}{log_prefix} CCXT lacks '{endpoint}'.{Style.RESET_ALL}"
            )
            return False

        params_switch = {
            "category": category,
            "symbol": market["id"],
            "tradeMode": 1,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        logger.debug(f"{log_prefix} Calling {endpoint} with params: {params_switch}")
        response = getattr(exchange, endpoint)(params_switch)
        logger.debug(f"{log_prefix} Raw V5 switch response: {response}")
        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "").lower()

        if ret_code == 0:
            logger.success(
                f"{Fore.GREEN}{log_prefix} Switched {symbol} to ISOLATED with {leverage}x leverage via V5.{Style.RESET_ALL}"
            )
            return True
        # Code 110026: Margin mode is not modified (already isolated)
        # Code 34036: Already in the target mode (UTA?)
        elif ret_code in [110026, 34036] or "margin mode is not modified" in ret_msg:
            logger.info(
                f"{Fore.CYAN}{log_prefix} {symbol} already ISOLATED via V5 check. Confirming leverage...{Style.RESET_ALL}"
            )
            # Explicitly call set_leverage again to ensure the leverage value is correct
            leverage_confirm_success = set_leverage(exchange, symbol, leverage, config)
            if leverage_confirm_success:
                logger.success(
                    f"{Fore.GREEN}{log_prefix} Leverage confirmed/set {leverage}x for ISOLATED {symbol}.{Style.RESET_ALL}"
                )
                return True
            else:
                logger.error(
                    f"{Fore.RED}{log_prefix} Failed leverage confirm/set after ISOLATED check.{Style.RESET_ALL}"
                )
                return False
        # Code 110020: Cannot switch mode with active positions/orders
        elif (
            ret_code == 110020
            or "have position" in ret_msg
            or "active order" in ret_msg
        ):
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot switch {symbol} to ISOLATED: active pos/orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"
            )
            return False
        else:
            # Raise unexpected V5 error codes
            raise ccxt.ExchangeError(
                f"Bybit API error switching isolated mode (V5): Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'"
            )

    except (
        ccxt.NetworkError,
        ccxt.ExchangeError,
        ccxt.AuthenticationError,
        ccxt.BadSymbol,
        ValueError,
    ) as e:
        # Don't log expected "have position" error loudly again
        if not (isinstance(e, ccxt.ExchangeError) and ret_code == 110020):
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} API/Input Error setting isolated margin: {e}{Style.RESET_ALL}"
            )
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)):
            raise e  # Allow retry
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error setting isolated margin: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


# --- Example Standalone Testing Block ---
if __name__ == "__main__":
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}--- Bybit V5 Helpers Module Standalone Execution ---{Style.RESET_ALL}"
    )
    print(
        "Basic syntax checks only. Depends on external Config, logger, and bybit_utils.py."
    )
    # List defined functions (excluding internal ones starting with _)
    all_funcs = [
        name
        for name, obj in locals().items()
        if callable(obj)
        and not name.startswith("_")
        and name not in ["Config", "AppConfig"]
    ]
    print(f"Found {len(all_funcs)} function definitions.")
    # Example: print(all_funcs)
    print(f"\n{Fore.GREEN}Basic syntax check passed.{Style.RESET_ALL}")
    print(
        f"Ensure PANDAS_AVAILABLE={PANDAS_AVAILABLE}, CCXT_AVAILABLE={CCXT_AVAILABLE}"
    )

# --- END OF FILE bybit_helper_functions.py ---
