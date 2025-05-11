# File: position_management.py
# -*- coding: utf-8 -*-

"""
Functions for Fetching, Closing, and Analyzing Positions on Bybit V5
"""

import logging
import sys
from decimal import Decimal
from typing import Optional, Dict, Any, Literal

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print(
        "Warning: colorama library not found. Logs will not be colored. Install: pip install colorama"
    )

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()

from config import Config
from utils import (
    retry_api_call,
    _get_v5_category,
    safe_decimal_conversion,
    format_price,
    format_amount,
    format_order_id,
    send_sms_alert,
)
# Import order function needed for closing position

logger = logging.getLogger(__name__)


# Snippet 8 / Function 8: Fetch Current Position (Bybit V5 Specific)
@retry_api_call(max_retries=3, initial_delay=1.0)
def get_current_position_bybit_v5(
    exchange: ccxt.bybit, symbol: str, config: Config
) -> Dict[str, Any]:
    """
    Fetches the current position details for a specific symbol using Bybit V5's fetchPositions logic.
    Focuses on One-Way position mode (positionIdx=0) by default.
    Returns a dictionary with Decimals for numeric values and a standardized structure.

    Returns (default if no position):
        {
            'symbol': symbol,
            'side': config.POS_NONE ('NONE', 'LONG', or 'SHORT'),
            'qty': Decimal("0.0"),
            'entry_price': Decimal("0.0"),
            'liq_price': None | Decimal,
            'mark_price': None | Decimal,
            'pnl_unrealized': None | Decimal,
            'leverage': None | Decimal,
            'info': {} # Raw position info from CCXT/exchange
        }
    """
    func_name = "get_current_position_bybit_v5"
    logger.debug(f"[{func_name}] Fetching position for {symbol} (Bybit V5)...")

    # Define the default structure for returning when no position exists
    default_position: Dict[str, Any] = {
        "symbol": symbol,
        "side": config.POS_NONE,
        "qty": Decimal("0.0"),
        "entry_price": Decimal("0.0"),
        "liq_price": None,
        "mark_price": None,
        "pnl_unrealized": None,
        "leverage": None,
        "info": {},
    }

    try:
        market = exchange.market(symbol)
        market_id = market["id"]  # Use market ID for V5 API calls
        category = _get_v5_category(market)

        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot fetch position for non-contract symbol: {symbol} (Category: {category}).{Style.RESET_ALL}"
            )
            return default_position

        # Check if fetchPositions is supported (should be for Bybit V5 in recent CCXT)
        if not hasattr(exchange, "fetch_positions") or not exchange.has.get(
            "fetchPositions"
        ):
            logger.error(
                f"{Fore.RED}[{func_name}] Exchange object does not support 'fetchPositions'. Cannot get position.{Style.RESET_ALL}"
            )
            return default_position

        # V5 fetchPositions requires category and optionally symbol
        params = {"category": category, "symbol": market_id}
        logger.debug(f"[{func_name}] Calling fetch_positions with params: {params}")

        # Fetch positions - CCXT might return multiple entries even for one symbol in hedge mode
        # We filter for the specific symbol and positionIdx=0 (One-Way)
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)

        active_position_data: Optional[Dict] = None
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            pos_symbol = pos_info.get("symbol")
            pos_v5_side = pos_info.get("side", "None")  # Bybit uses 'Buy'/'Sell'/'None'
            pos_size_str = pos_info.get("size")
            pos_idx = int(pos_info.get("positionIdx", -1))  # Default to -1 if not found

            # Match market ID, ensure side is not 'None', and check for One-Way index (0)
            if pos_symbol == market_id and pos_v5_side != "None" and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                # Check if size is significant (above epsilon)
                if abs(pos_size) > config.POSITION_QTY_EPSILON:
                    active_position_data = pos
                    logger.debug(
                        f"[{func_name}] Found active One-Way (idx 0) position data for {symbol}."
                    )
                    break  # Found the one-way position, stop searching

        # If an active one-way position was found, parse it
        if active_position_data:
            try:
                info = active_position_data.get("info", {})
                size = safe_decimal_conversion(info.get("size"))
                entry_price = safe_decimal_conversion(
                    info.get("avgPrice")
                )  # V5 uses avgPrice
                liq_price = safe_decimal_conversion(info.get("liqPrice"))
                mark_price = safe_decimal_conversion(info.get("markPrice"))
                pnl = safe_decimal_conversion(info.get("unrealisedPnl"))
                leverage = safe_decimal_conversion(info.get("leverage"))

                # Determine standardized side ('LONG', 'SHORT', 'NONE')
                pos_side_str = info.get("side")  # 'Buy' or 'Sell'
                position_side = (
                    config.POS_LONG
                    if pos_side_str == "Buy"
                    else (
                        config.POS_SHORT if pos_side_str == "Sell" else config.POS_NONE
                    )
                )

                # Quantity should always be positive, side indicates direction
                quantity = abs(size) if size is not None else Decimal("0.0")

                # Final check if parsed position is valid
                if (
                    position_side == config.POS_NONE
                    or quantity <= config.POSITION_QTY_EPSILON
                ):
                    logger.info(
                        f"[{func_name}] Parsed position for {symbol} has negligible size or side 'None'. Treating as flat."
                    )
                    return default_position

                # Log the found position details
                log_color = Fore.GREEN if position_side == config.POS_LONG else Fore.RED
                logger.info(
                    f"{log_color}[{func_name}] ACTIVE {position_side} Position {symbol}:{Style.RESET_ALL}"
                )
                logger.info(f"  Qty: {format_amount(exchange, symbol, quantity)}")
                logger.info(
                    f"  Entry Price: {format_price(exchange, symbol, entry_price)}"
                )
                logger.info(
                    f"  Mark Price: {format_price(exchange, symbol, mark_price)}"
                )
                logger.info(
                    f"  Liq. Price: {format_price(exchange, symbol, liq_price)}"
                )
                logger.info(
                    f"  Unrealized PNL: {format_price(exchange, config.USDT_SYMBOL, pnl)} {config.USDT_SYMBOL}"
                )
                logger.info(f"  Leverage: {leverage}x")

                return {
                    "symbol": symbol,
                    "side": position_side,
                    "qty": quantity,
                    "entry_price": entry_price,
                    "liq_price": liq_price,
                    "mark_price": mark_price,
                    "pnl_unrealized": pnl,
                    "leverage": leverage,
                    "info": info,  # Include the raw info dict
                }
            except Exception as parse_err:
                logger.warning(
                    f"{Fore.YELLOW}[{func_name}] Error parsing details from active position data: {parse_err}. Data: {str(active_position_data)[:300]}{Style.RESET_ALL}"
                )
                return default_position  # Return default on parsing error
        else:
            # No active one-way position found in the fetched data
            logger.info(
                f"[{func_name}] No active One-Way position found for {symbol} after checking fetched data."
            )
            return default_position

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] API Error fetching position for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching position for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return default_position  # Return default on unexpected errors


# Snippet 9 / Function 9: Close Position (Reduce-Only Market)
@retry_api_call(
    max_retries=2, initial_delay=1
)  # Allow retry for closure attempt if network fails
def close_position_reduce_only(
    exchange: ccxt.bybit,
    symbol: str,
    config: Config,
    position_to_close: Optional[Dict[str, Any]] = None,
    reason: str = "Signal Close",
) -> Optional[Dict[str, Any]]:
    """
    Closes the current position for the given symbol using a reduce-only market order.
    Optionally takes existing position data to avoid an extra fetch.
    Handles specific "already closed" or "order would not reduce" exchange errors gracefully.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol to close the position for.
        config: Configuration object.
        position_to_close: Optional pre-fetched position dictionary (from get_current_position_bybit_v5).
                           If None, the function will fetch the current position first.
        reason: A string indicating the reason for closing (for logging).

    Returns:
        The market order dictionary returned by ccxt if the close order was successfully placed,
        None if there was no position to close, if the position was already closed (based on errors),
        or if the close order placement failed.
    """
    func_name = "close_position_reduce_only"
    market_base = symbol.split("/")[0]
    log_prefix = f"[Close Position ({reason})]"
    logger.info(
        f"{Fore.YELLOW}{log_prefix} Initiating closure for {symbol}...{Style.RESET_ALL}"
    )

    live_position_data: Dict[str, Any]

    # 1. Determine current position state
    if (
        position_to_close
        and isinstance(position_to_close, dict)
        and "side" in position_to_close
        and "qty" in position_to_close
    ):
        logger.debug(f"[{func_name}] Using provided position state.")
        live_position_data = position_to_close
    else:
        logger.debug(f"[{func_name}] Fetching current position state for {symbol}...")
        try:
            live_position_data = get_current_position_bybit_v5(exchange, symbol, config)
        except Exception as fetch_err:
            # If fetching position fails, we cannot proceed safely
            logger.error(
                f"{Fore.RED}{log_prefix} Failed to fetch current position state for {symbol}: {fetch_err}. Aborting close.{Style.RESET_ALL}"
            )
            return None

    # Extract side and quantity
    live_side = live_position_data.get("side", config.POS_NONE)
    live_qty = live_position_data.get("qty", Decimal("0.0"))

    # 2. Validate if there is an active position to close
    if live_side == config.POS_NONE or live_qty <= config.POSITION_QTY_EPSILON:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] No active position found or quantity is zero for {symbol}. Nothing to close.{Style.RESET_ALL}"
        )
        return None  # Nothing to close

    # 3. Determine the side required for the closing market order
    close_order_side: Literal["buy", "sell"]
    if live_side == config.POS_LONG:
        close_order_side = config.SIDE_SELL
    elif live_side == config.POS_SHORT:
        close_order_side = config.SIDE_BUY
    else:
        # Should have been caught above, but safeguard
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid live position side '{live_side}' detected. Aborting close.{Style.RESET_ALL}"
        )
        return None

    # 4. Place the Reduce-Only Market Order
    try:
        # Use the dedicated market order function, ensuring reduceOnly=True
        # Pass the exact live_qty to close the full position
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Attempting to close {live_side} position of {format_amount(exchange, symbol, live_qty)} {symbol} "
            f"by placing a {close_order_side.upper()} MARKET order (ReduceOnly).{Style.RESET_ALL}"
        )

        # Use place_market_order_slippage_check or a simpler create_market_order call
        # Using the slippage check function here adds safety but also latency/complexity
        # For simplicity, let's use a direct create_market_order call with reduceOnly param
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            raise ValueError("Cannot determine category for close order.")

        qty_str = format_amount(exchange, symbol, live_qty)
        qty_float = float(qty_str)
        params: Dict[str, Any] = {"category": category, "reduceOnly": True}

        bg = Back.YELLOW
        fg = Fore.BLACK
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}[{func_name}] Executing CLOSE {live_side} ({reason}): "
            f"{close_order_side.upper()} MARKET {qty_str} {symbol} (ReduceOnly)...{Style.RESET_ALL}"
        )

        close_order = exchange.create_market_order(
            symbol=symbol, side=close_order_side, amount=qty_float, params=params
        )

        if not close_order:
            # This case might occur if the API call itself fails without raising ccxt exception
            raise ValueError(
                "create_market_order returned None unexpectedly during close attempt."
            )

        # 5. Log Success and Return Order Info
        fill_price = safe_decimal_conversion(close_order.get("average"))
        fill_qty = safe_decimal_conversion(close_order.get("filled", "0.0"))
        order_cost = safe_decimal_conversion(close_order.get("cost", "0.0"))
        order_id = format_order_id(close_order.get("id"))
        status = close_order.get("status", "?")

        # Use logger.info for success
        logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}[{func_name}] Close Order ({reason}) submitted for {symbol}. "
            f"ID:...{order_id}, Status:{status}, Filled:{format_amount(exchange, symbol, fill_qty)}/{qty_str}, "
            f"AvgFill:{format_price(exchange, symbol, fill_price)}, Cost:{order_cost:.4f}{Style.RESET_ALL}"
        )

        send_sms_alert(
            f"[{market_base}] Closed {live_side} {qty_str} @ ~{format_price(exchange, symbol, fill_price)} ({reason}). ID:...{order_id}",
            config,
        )
        return close_order

    # 6. Handle Specific Errors Gracefully
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
        # These are likely real issues preventing closure
        logger.error(
            f"{Fore.RED}[{func_name}] Close Order Error ({reason}) for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}] CLOSE FAIL ({live_side}): {type(e).__name__}", config
        )
        # Decide whether to raise or return None. Raising might trigger outer retries.
        # raise e
        return None
    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        # Check for common Bybit V5 error codes/messages indicating the position is already closed or the order wouldn't reduce
        # Codes: 110025 (Position is closed), 110045 (Order would not reduce position), 30086 (position size is zero) - check Bybit docs for updates
        already_closed_indicators = [
            "110025",
            "110045",
            "30086",
            "position is closed",
            "order would not reduce",
            "position size is zero",
        ]
        if any(indicator in error_str for indicator in already_closed_indicators):
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Close Order ({reason}) for {symbol}: Exchange indicates position already closed or order would not reduce: {e}. Assuming closed.{Style.RESET_ALL}"
            )
            return None  # Treat as success (or non-action needed)
        else:
            # Other exchange errors are more problematic
            logger.error(
                f"{Fore.RED}[{func_name}] Close Order ExchangeError ({reason}) for {symbol}: {e}{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"[{market_base}] CLOSE FAIL ({live_side}): ExchangeError - {e}", config
            )
            raise e  # Re-raise other exchange errors for potential retry or handling
    except (ccxt.NetworkError, ValueError) as e:
        # Network errors or value errors (e.g., from category determination)
        logger.error(
            f"{Fore.RED}[{func_name}] Close Order Network/Setup Error ({reason}) for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise e  # Re-raise for potential retry
    except Exception as e:
        # Catch-all for unexpected issues
        logger.critical(
            f"{Back.RED}[{func_name}] Close Order Unexpected Error ({reason}) for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{market_base}] CLOSE FAIL ({live_side}): Unexpected Error - {type(e).__name__}",
            config,
        )
        return None


# Snippet 23 / Function 23: Fetch Position Risk (Bybit V5 Specific)
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_position_risk_bybit_v5(
    exchange: ccxt.bybit, symbol: str, config: Config
) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed risk metrics for the current position of a specific symbol using Bybit V5 logic.
    Attempts to use `fetch_positions_risk` first, falls back to `fetch_positions` if unavailable/fails.
    Focuses on One-Way mode (positionIdx=0).

    Returns a detailed dictionary with Decimal values for numeric fields, or None if no
    active One-Way position is found or an error occurs.

    Returned Dict Structure (example):
        {
            'symbol': str,
            'side': Literal['LONG', 'SHORT', 'NONE'],
            'qty': Decimal,
            'entry_price': Decimal | None,
            'mark_price': Decimal | None,
            'liq_price': Decimal | None,
            'leverage': Decimal | None,
            'initial_margin': Decimal | None, # Position Initial Margin (IM)
            'maint_margin': Decimal | None,   # Position Maintenance Margin (MM)
            'unrealized_pnl': Decimal | None,
            'imr': Decimal | None,            # Initial Margin Rate
            'mmr': Decimal | None,            # Maintenance Margin Rate
            'position_value': Decimal | None, # Value of the position at mark price
            'risk_limit_value': Decimal | None, # Current risk limit value used
            'info': Dict                    # Raw info dictionary from CCXT
        }
    """
    func_name = "fetch_position_risk_bybit_v5"
    logger.debug(
        f"[{func_name}] Fetching detailed position risk for {symbol} (Bybit V5)..."
    )

    try:
        market = exchange.market(symbol)
        market_id = market["id"]  # Use market ID for V5 API calls
        category = _get_v5_category(market)

        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot fetch position risk for non-contract symbol: {symbol} (Category: {category}).{Style.RESET_ALL}"
            )
            return None

        params = {"category": category, "symbol": market_id}
        position_data: Optional[List[Dict]] = None
        fetch_method_used = "N/A"

        # Try fetchPositionsRisk first (preferred as it's specific)
        if exchange.has.get("fetchPositionsRisk"):
            try:
                logger.debug(
                    f"[{func_name}] Attempting fetch_positions_risk with params: {params}"
                )
                position_data = exchange.fetch_positions_risk(
                    symbols=[symbol], params=params
                )
                fetch_method_used = "fetchPositionsRisk"
            except Exception as e:
                logger.warning(
                    f"[{func_name}] fetch_positions_risk failed ({type(e).__name__}: {e}). Falling back to fetch_positions."
                )
                position_data = None  # Ensure fallback occurs
        else:
            logger.debug(
                f"[{func_name}] fetch_positions_risk not available in this CCXT version."
            )

        # Fallback to fetchPositions if fetchPositionsRisk failed or is unavailable
        if position_data is None:
            if exchange.has.get("fetchPositions"):
                logger.debug(
                    f"[{func_name}] Attempting fallback fetch_positions with params: {params}"
                )
                try:
                    position_data = exchange.fetch_positions(
                        symbols=[symbol], params=params
                    )
                    fetch_method_used = "fetchPositions (Fallback)"
                except Exception as e:
                    logger.error(
                        f"[{func_name}] Fallback fetch_positions also failed ({type(e).__name__}: {e}). Cannot get position data."
                    )
                    position_data = None  # Ensure it remains None
            else:
                logger.error(
                    f"{Fore.RED}[{func_name}] Neither fetchPositionsRisk nor fetchPositions are available. Cannot get position risk.{Style.RESET_ALL}"
                )
                return None  # Cannot proceed

        if position_data is None:
            # This means both fetch attempts (if applicable) failed without raising exceptions handled by retry
            logger.error(
                f"{Fore.RED}[{func_name}] Failed to fetch position data using available methods ({fetch_method_used}).{Style.RESET_ALL}"
            )
            return None

        # Find the active One-Way position (idx 0) within the results
        active_pos_risk: Optional[Dict] = None
        for pos in position_data:
            pos_info = pos.get("info", {})
            pos_symbol = pos_info.get("symbol")
            pos_v5_side = pos_info.get(
                "side", "None"
            )  # Bybit side: 'Buy', 'Sell', 'None'
            pos_size_str = pos_info.get("size")
            pos_idx = int(pos_info.get("positionIdx", -1))

            if pos_symbol == market_id and pos_v5_side != "None" and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str)
                # Check if position size is significant
                if pos_size is not None and abs(pos_size) > config.POSITION_QTY_EPSILON:
                    active_pos_risk = pos
                    logger.debug(
                        f"[{func_name}] Found active One-Way position risk data for {symbol} using {fetch_method_used}."
                    )
                    break

        if not active_pos_risk:
            logger.info(
                f"[{func_name}] No active One-Way position found for {symbol} in the fetched data."
            )
            return None  # Return None if no active position

        # Parse the details from the found position risk data
        try:
            info = active_pos_risk.get("info", {})

            # Extract common fields, checking both CCXT standard keys and V5 info keys
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
            # Margins
            initial_margin = safe_decimal_conversion(
                active_pos_risk.get("initialMargin", info.get("positionIM"))
            )  # positionIM in V5
            maint_margin = safe_decimal_conversion(
                active_pos_risk.get("maintenanceMargin", info.get("positionMM"))
            )  # positionMM in V5
            # PNL
            pnl = safe_decimal_conversion(
                active_pos_risk.get("unrealizedPnl", info.get("unrealisedPnl"))
            )
            # Margin Rates
            imr = safe_decimal_conversion(
                active_pos_risk.get("initialMarginPercentage", info.get("imr"))
            )  # imr in V5
            mmr = safe_decimal_conversion(
                active_pos_risk.get("maintenanceMarginPercentage", info.get("mmr"))
            )  # mmr in V5
            # Other info
            pos_value = safe_decimal_conversion(
                active_pos_risk.get("contractsValue", info.get("positionValue"))
            )  # positionValue in V5
            risk_limit = safe_decimal_conversion(
                info.get("riskLimitValue")
            )  # V5 specific

            # Determine standardized side
            pos_side_str = info.get("side")  # 'Buy' or 'Sell'
            position_side = (
                config.POS_LONG
                if pos_side_str == "Buy"
                else (config.POS_SHORT if pos_side_str == "Sell" else config.POS_NONE)
            )

            # Quantity (absolute value)
            quantity = abs(size) if size is not None else Decimal("0.0")

            # Final validation on parsed data
            if (
                position_side == config.POS_NONE
                or quantity <= config.POSITION_QTY_EPSILON
            ):
                logger.info(
                    f"[{func_name}] Parsed position risk for {symbol} resulted in negligible size or side 'None'."
                )
                return None

            # Log detailed risk information
            log_color = Fore.GREEN if position_side == config.POS_LONG else Fore.RED
            quote_curr = market.get("quote", config.USDT_SYMBOL)
            logger.info(
                f"{log_color}[{func_name}] Position Risk Details {symbol} ({position_side}):{Style.RESET_ALL}"
            )
            logger.info(
                f"  Qty: {format_amount(exchange, symbol, quantity)}, Entry: {format_price(exchange, symbol, entry_price)}, Mark: {format_price(exchange, symbol, mark_price)}"
            )
            logger.info(
                f"  Liq. Price: {format_price(exchange, symbol, liq_price)}, Leverage: {leverage}x"
            )
            logger.info(
                f"  Position Value: {format_price(exchange, quote_curr, pos_value)} {quote_curr}"
            )
            logger.info(
                f"  Unrealized PNL: {format_price(exchange, quote_curr, pnl)} {quote_curr}"
            )
            logger.info(
                f"  Initial Margin (IM): {format_price(exchange, quote_curr, initial_margin)} {quote_curr} (IMR: {imr:.4% if imr else 'N/A'})"
            )
            logger.info(
                f"  Maint. Margin (MM): {format_price(exchange, quote_curr, maint_margin)} {quote_curr} (MMR: {mmr:.4% if mmr else 'N/A'})"
            )
            logger.info(f"  Current Risk Limit Value: {risk_limit or 'N/A'}")

            # Return the structured dictionary
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
                "info": info,  # Include raw info
            }
        except Exception as parse_err:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Error parsing position risk details: {parse_err}. Data: {str(active_pos_risk)[:300]}{Style.RESET_ALL}"
            )
            return None  # Return None if parsing fails

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] API Error fetching position risk for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching position risk for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- END OF FILE position_management.py ---

# ---------------------------------------------------------------------------
