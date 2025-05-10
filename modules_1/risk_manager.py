# File: risk_manager.py
import logging
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Dict, Optional, Union

import ccxt

# Import utility functions
from utils import (
    NEON_GREEN,
    RESET_ALL_STYLE as RESET,
    get_price_precision,
)  # Corrected import and aliased RESET_ALL_STYLE


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: Union[float, Decimal, str],
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange,
    logger: Optional[logging.Logger] = None,
) -> Optional[Decimal]:
    """
    Calculates the position size in base currency or contracts.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
    quote_currency = market_info.get("quote", "QUOTE")
    base_currency = market_info.get("base", "BASE")
    is_contract = market_info.get("is_contract", False)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation and Conversion ---
    if not (isinstance(balance, Decimal) and balance > 0):
        lg.error(
            f"Position sizing failed ({symbol}): Invalid or non-positive balance ({balance})."
        )
        return None

    try:
        risk_value_decimal = Decimal(str(risk_per_trade))
        if not (Decimal(0) <= risk_value_decimal <= Decimal(1)):
            raise ValueError(
                f"risk_per_trade ({risk_value_decimal}) must be between 0 and 1 (e.g., 0.01 for 1%)."
            )
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(
            f"Position sizing failed ({symbol}): Invalid risk_per_trade ('{risk_per_trade}'). Error: {e}"
        )
        return None

    if not (
        isinstance(initial_stop_loss_price, Decimal) and initial_stop_loss_price > 0
    ):
        lg.error(
            f"Position sizing failed ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price})."
        )
        return None
    if not (isinstance(entry_price, Decimal) and entry_price > 0):
        lg.error(
            f"Position sizing failed ({symbol}): Invalid entry_price ({entry_price})."
        )
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(
            f"Position sizing failed ({symbol}): Stop loss price cannot be equal to entry price."
        )
        return None
    if "limits" not in market_info or "precision" not in market_info:
        lg.error(
            f"Position sizing failed ({symbol}): Market info missing 'limits' or 'precision'."
        )
        return None

    try:
        risk_amount_quote = balance * risk_value_decimal
        if risk_value_decimal > 0 and risk_amount_quote <= 0:
            lg.warning(
                f"Calculated risk amount {risk_amount_quote} {quote_currency} for {symbol} (risk {risk_value_decimal:.2%}) is non-positive. Balance: {balance:.4f}"
            )

        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
            lg.error(
                f"Position sizing failed ({symbol}): Stop loss distance non-positive ({sl_distance_per_unit})."
            )
            return None

        contract_size_str = market_info.get("contractSize", "1")
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0:
                contract_size = Decimal("1")
                lg.warning(f"Contract size invalid for {symbol}, defaulted to 1.")
        except (InvalidOperation, ValueError, TypeError):
            contract_size = Decimal("1")
            lg.warning(f"Contract size parse error for {symbol}, defaulted to 1.")

        calculated_size: Optional[Decimal] = None
        if market_info.get("linear", True) or not is_contract:
            denominator = sl_distance_per_unit * contract_size
            if denominator > 0:
                calculated_size = risk_amount_quote / denominator
            else:
                lg.error(
                    f"Pos sizing denom err ({symbol}): SLDist={sl_distance_per_unit}, ContrSize={contract_size}."
                )
                return None
        elif market_info.get("inverse", False):
            if entry_price == 0 or initial_stop_loss_price == 0:
                lg.error(
                    f"Pos sizing err ({symbol}): Entry or SL price zero for inverse contract."
                )
                return None
            loss_per_contract_in_quote = contract_size * abs(
                Decimal("1") / entry_price - Decimal("1") / initial_stop_loss_price
            )
            if loss_per_contract_in_quote > 0:
                calculated_size = risk_amount_quote / loss_per_contract_in_quote
            else:
                lg.error(
                    f"Pos sizing err ({symbol}): Loss per contract zero/neg for inverse."
                )
                return None
        else:
            lg.error(
                f"Unsupported market type for sizing {symbol}. Market: {market_info.get('type')}"
            )
            return None

        if not (calculated_size and calculated_size > 0):
            lg.error(
                f"Initial size calc zero/neg: {calculated_size}. RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_per_unit}, ContrSize={contract_size}"
            )
            return None

        # Use get_price_precision for formatting the log output
        price_precision_for_log = get_price_precision(market_info, lg)
        lg.info(
            f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_value_decimal:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}"
        )
        lg.info(
            f"  Entry={entry_price:.{price_precision_for_log}f}, SL={initial_stop_loss_price:.{price_precision_for_log}f}, SL Dist Per Unit={sl_distance_per_unit:.{price_precision_for_log}f}"
        )
        lg.info(
            f"  ContractSize={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}"
        )

        limits = market_info.get("limits", {})
        amount_limits = limits.get("amount", {}) if isinstance(limits, dict) else {}
        cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}

        min_amount_str = amount_limits.get("min")
        max_amount_str = amount_limits.get("max")
        min_cost_str = cost_limits.get("min")
        max_cost_str = cost_limits.get("max")

        min_amount = Decimal("0")
        if min_amount_str is not None and str(min_amount_str).strip():
            try:
                val = Decimal(str(min_amount_str))
                if val >= 0:
                    min_amount = val  # Allow 0 min_amount
            except InvalidOperation:
                lg.warning(
                    f"Invalid min_amount_str '{min_amount_str}' for {symbol}. Using 0."
                )
                min_amount = Decimal("0")

        max_amount: Optional[Decimal] = None
        if max_amount_str is not None and str(max_amount_str).strip():
            try:
                val = Decimal(str(max_amount_str))
                if val > 0:
                    max_amount = val
            except InvalidOperation:
                lg.warning(
                    f"Invalid max_amount_str '{max_amount_str}' for {symbol}. No upper amount limit."
                )

        min_cost = Decimal("0")
        if min_cost_str is not None and str(min_cost_str).strip():
            try:
                val = Decimal(str(min_cost_str))
                if val >= 0:
                    min_cost = val  # Allow 0 min_cost
            except InvalidOperation:
                lg.warning(
                    f"Invalid min_cost_str '{min_cost_str}' for {symbol}. Using 0."
                )
                min_cost = Decimal("0")

        max_cost: Optional[Decimal] = None
        if max_cost_str is not None and str(max_cost_str).strip():
            try:
                val = Decimal(str(max_cost_str))
                if val > 0:
                    max_cost = val
            except InvalidOperation:
                lg.warning(
                    f"Invalid max_cost_str '{max_cost_str}' for {symbol}. No upper cost limit."
                )

        adjusted_size = calculated_size
        if min_amount > 0 and adjusted_size < min_amount:
            lg.warning(
                f"Calculated size {calculated_size:.8f} < min amount {min_amount}. Adjusting to min for {symbol}."
            )
            adjusted_size = min_amount
        if max_amount and adjusted_size > max_amount:
            lg.warning(
                f"Calculated size {calculated_size:.8f} > max amount {max_amount}. Adjusting to max for {symbol}."
            )
            adjusted_size = max_amount

        estimated_cost: Optional[Decimal] = None
        cost_calc_price = entry_price
        if market_info.get("linear", True) or not is_contract:
            if cost_calc_price > 0 and contract_size > 0:
                estimated_cost = adjusted_size * cost_calc_price * contract_size
        elif market_info.get("inverse", False):
            if cost_calc_price > 0 and contract_size > 0:
                estimated_cost = adjusted_size * contract_size / cost_calc_price

        if estimated_cost is None:
            lg.error(
                f"Could not estimate cost for {symbol} (price or contract size invalid)."
            )
            return None
        lg.debug(
            f"  Size after amount limits: {adjusted_size:.8f}. Est. Cost: {estimated_cost:.4f} {quote_currency}"
        )

        if min_cost > 0 and estimated_cost < min_cost:
            lg.warning(
                f"Est. cost {estimated_cost:.4f} < min_cost {min_cost} for {symbol}. Trying to meet min_cost."
            )
            required_size_for_min_cost: Optional[Decimal] = None

            cost_per_unit_denom: Optional[Decimal] = None
            if market_info.get("linear", True) or not is_contract:
                cost_per_unit_denom = cost_calc_price * contract_size
            elif market_info.get("inverse", False) and cost_calc_price > 0:
                # For inverse: Cost = Size * ContractSize / Price => Size = Cost * Price / ContractSize
                if contract_size > 0:
                    required_size_for_min_cost = (
                        min_cost * cost_calc_price
                    ) / contract_size
                cost_per_unit_denom = None  # Flag that required_size_for_min_cost was calculated differently

            if (
                cost_per_unit_denom and cost_per_unit_denom > 0
            ):  # For linear/spot if not already calculated
                required_size_for_min_cost = min_cost / cost_per_unit_denom

            if not (required_size_for_min_cost and required_size_for_min_cost > 0):
                lg.error(
                    f"Cannot calc required size for min_cost for {symbol} (denom/price invalid)."
                )
                return None
            lg.info(
                f"  Required size for min_cost {min_cost}: {required_size_for_min_cost:.8f} {size_unit}"
            )

            if max_amount and required_size_for_min_cost > max_amount:
                lg.error(
                    f"Cannot meet min_cost for {symbol} without exceeding max_amount. Aborted."
                )
                return None
            if (
                min_amount > 0 and required_size_for_min_cost < min_amount
            ):  # Check against actual min_amount, not calculated_size
                lg.error(
                    f"Req. size for min_cost {required_size_for_min_cost} < min_amount {min_amount}. Limits conflict for {symbol}. Aborted."
                )
                return None
            adjusted_size = required_size_for_min_cost
            lg.info(
                f"  Adjusted size to meet min_cost: {adjusted_size:.8f} {size_unit}"
            )
            # Recalculate estimated_cost for max_cost check
            if market_info.get("linear", True) or not is_contract:
                estimated_cost = adjusted_size * cost_calc_price * contract_size
            elif market_info.get("inverse", False) and cost_calc_price > 0:
                estimated_cost = adjusted_size * contract_size / cost_calc_price
            if estimated_cost is None:
                lg.error(f"Could not re-estimate cost for {symbol} after min_cost adj.")
                return None

        if max_cost and estimated_cost > max_cost:
            lg.warning(
                f"Est. cost {estimated_cost:.4f} > max_cost {max_cost} for {symbol}. Reducing size."
            )
            size_for_max_cost: Optional[Decimal] = None
            cost_per_unit_denom_mc: Optional[Decimal] = None
            if market_info.get("linear", True) or not is_contract:
                cost_per_unit_denom_mc = cost_calc_price * contract_size
                if cost_per_unit_denom_mc and cost_per_unit_denom_mc > 0:
                    size_for_max_cost = max_cost / cost_per_unit_denom_mc
            elif (
                market_info.get("inverse", False)
                and cost_calc_price > 0
                and contract_size > 0
            ):
                size_for_max_cost = (max_cost * cost_calc_price) / contract_size

            if not (size_for_max_cost and size_for_max_cost > 0):
                lg.error(f"Cannot calc max size for max_cost for {symbol}.")
                return None
            lg.info(f"  Reduced size by max_cost: {size_for_max_cost:.8f} {size_unit}")

            if min_amount > 0 and size_for_max_cost < min_amount:
                lg.error(
                    f"Size reduced for max_cost {size_for_max_cost} < min_amount {min_amount}. Aborted for {symbol}."
                )
                return None
            adjusted_size = size_for_max_cost

        final_size_str: Optional[str] = None
        try:
            # CCXT's amount_to_precision expects a float or string number as its second argument.
            # Passing Decimal directly might work for some exchanges but float is safer for CCXT's internal handling.
            final_size_str = exchange.amount_to_precision(symbol, float(adjusted_size))
            final_size = Decimal(final_size_str)
            lg.info(
                f"Applied exchange amount precision: {adjusted_size:.8f} -> {final_size} {size_unit}"
            )
        except Exception as e_ccxt_prec:
            lg.error(
                f"Error applying CCXT amount_to_precision for {symbol} on size {adjusted_size}: {e_ccxt_prec}. Attempting manual quantization."
            )
            amount_precision_places = market_info.get("amountPrecision", 8)
            quant_factor = Decimal("1e-" + str(amount_precision_places))
            final_size = adjusted_size.quantize(quant_factor, rounding=ROUND_DOWN)
            lg.info(
                f"Applied manual amount quantization: {adjusted_size:.8f} -> {final_size} {size_unit}"
            )

        if not (final_size and final_size > 0):
            lg.error(
                f"Final position size zero or negative ({final_size}) after all adjustments for {symbol}. Aborted."
            )
            return None
        if min_amount > 0 and final_size < min_amount:
            lg.error(
                f"Final size {final_size} < min amount {min_amount} after precision for {symbol}. Aborted."
            )
            return None

        final_cost_est: Optional[Decimal] = None
        cost_calc_price_final = (
            entry_price  # Use the same price for final cost check consistency
        )
        if market_info.get("linear", True) or not is_contract:
            if cost_calc_price_final > 0 and contract_size > 0:
                final_cost_est = final_size * cost_calc_price_final * contract_size
        elif market_info.get("inverse", False):
            if cost_calc_price_final > 0 and contract_size > 0:
                final_cost_est = final_size * contract_size / cost_calc_price_final

        if final_cost_est is not None and min_cost > 0 and final_cost_est < min_cost:
            lg.error(
                f"Final size {final_size} results in cost {final_cost_est:.4f} < min_cost {min_cost} for {symbol}. Aborted."
            )
            return None

        lg.info(
            f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}"
        )
        return final_size

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(
            f"Error during position size calculation ({symbol}) (Decimal/Type Error): {e}",
            exc_info=False,
        )
    except Exception as e:
        lg.error(
            f"Unexpected error calculating position size for {symbol}: {e}",
            exc_info=True,
        )
    return None
