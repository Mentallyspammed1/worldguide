# File: margin_calculation.py
# -*- coding: utf-8 -*-

"""
Functions for Calculating Margin Requirements
"""

import logging
import sys
from decimal import Decimal, DivisionByZero
from typing import Optional, Tuple, Literal

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored. Install: pip install colorama")
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

from config import Config
from utils import safe_decimal_conversion, format_price, format_amount

logger = logging.getLogger(__name__)


# Snippet 16 / Function 16: Calculate Margin Requirement
def calculate_margin_requirement(
    exchange: ccxt.bybit, symbol: str, amount: Decimal, price: Decimal, leverage: Decimal, config: Config,
    order_side: Literal['buy', 'sell'], is_maker: bool = False
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Calculates the estimated Initial Margin (IM) requirement for placing an order on Bybit V5.
    Also provides a basic Maintenance Margin (MM) estimate based on market MMR.

    Note: This provides an *estimate*. Actual margin used depends on the execution price,
    real-time MMR/IMR tiers, potential order fees, and account state (cross/isolated).

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        amount: Order quantity in base currency (Decimal). Must be positive.
        price: Estimated execution price for the order (Decimal). Must be positive.
        leverage: The leverage to be used for this order (Decimal). Must be positive.
        config: Configuration object (for fee rates, quote currency).
        order_side: 'buy' or 'sell'. (Currently not used in calc, but good for context).
        is_maker: True if the order is intended to be a maker order (affects fee estimate).

    Returns:
        A tuple containing (estimated_initial_margin, estimated_maintenance_margin) as Decimals.
        Returns (None, None) if calculation fails or inputs are invalid.
    """
    func_name = "calculate_margin_requirement"
    logger.debug(f"[{func_name}] Calculating margin requirement: {order_side} {format_amount(exchange, symbol, amount)} {symbol} "
                 f"@ {format_price(exchange, symbol, price)}, Leverage: {leverage}x, Is Maker: {is_maker}")

    # Input validation
    if amount <= Decimal("0") or price <= Decimal("0") or leverage <= Decimal("0"):
        logger.error(f"{Fore.RED}[{func_name}] Invalid input: Amount ({amount}), Price ({price}), or Leverage ({leverage}) must be positive.{Style.RESET_ALL}")
        return None, None

    try:
        market = exchange.market(symbol)
        if not market: # Should not happen if symbol is valid, but check anyway
             raise ValueError(f"Market data not found for symbol '{symbol}'")

        # Ensure it's a contract market where margin applies
        if not market.get('contract'):
            logger.error(f"{Fore.RED}[{func_name}] Cannot calculate margin for non-contract symbol: {symbol}.{Style.RESET_ALL}")
            return None, None

        quote_currency = market.get('quote', config.USDT_SYMBOL) # Default to config USDT if not found

        # Calculate Order Value (Position Value)
        # order_value = amount * price
        # Using CCXT's cost calculation method might be more robust if available/needed
        order_value = exchange.cost(symbol=symbol, amount=float(amount), price=float(price), side=order_side, takerOrMaker='maker' if is_maker else 'taker')
        order_value = safe_decimal_conversion(order_value)
        if order_value is None:
             logger.warning(f"[{func_name}] Could not use exchange.cost, falling back to amount*price")
             order_value = amount * price

        logger.debug(f"[{func_name}] Estimated Order Value: {format_price(exchange, quote_currency, order_value)} {quote_currency}")

        # Calculate Base Initial Margin (IM = Order Value / Leverage)
        if leverage == Decimal("0"): # Should be caught by initial check, but safeguard
            raise DivisionByZero("Leverage cannot be zero.")
        initial_margin_base = order_value / leverage
        logger.debug(f"[{func_name}] Base Initial Margin (Value / Leverage): {format_price(exchange, quote_currency, initial_margin_base)} {quote_currency}")

        # Estimate Fees (Taker or Maker)
        # Note: Actual fees might be tiered or have rebates. This is a basic estimate.
        fee_rate = config.MAKER_FEE_RATE if is_maker else config.TAKER_FEE_RATE
        estimated_fee = order_value * fee_rate
        logger.debug(f"[{func_name}] Estimated Fee (Rate: {fee_rate:.4%}): {format_price(exchange, quote_currency, estimated_fee)} {quote_currency}")

        # Total Estimated Initial Margin = Base IM + Estimated Fee
        # Bybit typically requires margin for the position value *plus* the potential closing fee.
        # Some interpretations add opening fee instead/as well. Adding opening fee here is safer estimate.
        total_initial_margin_estimate = initial_margin_base + estimated_fee
        logger.info(f"[{func_name}] Estimated TOTAL Initial Margin Required (Base IM + Est. Open Fee): "
                    f"{format_price(exchange, quote_currency, total_initial_margin_estimate)} {quote_currency}")

        # Estimate Maintenance Margin (MM = Order Value * MMR)
        maintenance_margin_estimate: Optional[Decimal] = None
        try:
            # Try to get MMR from market info (check common keys)
            mmr_rate_str = market.get('maintenanceMarginRate') or \
                           market.get('info', {}).get('maintainMargin') or \
                           market.get('info', {}).get('mmr') or \
                           market.get('info', {}).get('maintenanceMarginRate') # Check a few common names

            if mmr_rate_str:
                mmr_rate = safe_decimal_conversion(mmr_rate_str)
                if mmr_rate and mmr_rate >= 0: # MMR should be non-negative
                    # Basic MM = Order Value * MMR
                    # Note: Bybit uses tiered MMR based on position value. This uses the base rate.
                    maintenance_margin_estimate = order_value * mmr_rate
                    logger.debug(f"[{func_name}] Basic Maintenance Margin Estimate (using base MMR {mmr_rate:.4%}): "
                                 f"{format_price(exchange, quote_currency, maintenance_margin_estimate)} {quote_currency}")
                else:
                    logger.warning(f"[{func_name}] Could not parse valid MMR rate from market info: '{mmr_rate_str}'")
            else:
                logger.debug(f"[{func_name}] Maintenance Margin Rate (MMR) not found in market info for {symbol}.")

        except Exception as mm_err:
            logger.warning(f"[{func_name}] Could not estimate Maintenance Margin due to error: {mm_err}")

        return total_initial_margin_estimate, maintenance_margin_estimate

    except (DivisionByZero, KeyError, ValueError) as e:
        logger.error(f"{Fore.RED}[{func_name}] Calculation error: {e}{Style.RESET_ALL}")
        return None, None
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error during margin calculation for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return None, None


# --- END OF FILE margin_calculation.py ---

# ---------------------------------------------------------------------------

