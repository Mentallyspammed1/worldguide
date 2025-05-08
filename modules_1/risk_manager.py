```python
# File: risk_manager.py
import logging
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Dict, Optional, Union, Any

import ccxt

# Import utility functions
from utils import (
    NEON_GREEN,
    RESET_ALL_STYLE as RESET,
    get_price_precision,
)

# --- Constants for dictionary keys ---
# Market Info Keys
KEY_SYMBOL = "symbol"
KEY_QUOTE_CURRENCY = "quote"
KEY_BASE_CURRENCY = "base"
KEY_IS_CONTRACT = "is_contract"
KEY_CONTRACT_SIZE = "contractSize"
KEY_LINEAR = "linear"
KEY_INVERSE = "inverse"
KEY_TYPE = "type"  # For logging market type
KEY_LIMITS = "limits"
KEY_PRECISION_INFO = "precision" # CCXT 'precision' dict
KEY_AMOUNT_PRECISION_FALLBACK = "amountPrecision" # Custom fallback num decimal places

# Limits Keys
KEY_AMOUNT = "amount"
KEY_COST = "cost"
KEY_MIN = "min"
KEY_MAX = "max"

# Default logger if none provided
DEFAULT_LOGGER = logging.getLogger(__name__)


def _parse_decimal_limit(
    limit_value_str: Optional[Any],
    limit_name: str,
    symbol_for_log: str,
    logger_instance: logging.Logger,
    default_if_none_or_invalid: Decimal,
    is_max_limit: bool = False,
) -> Optional[Decimal]:
    """
    Parses a limit string (min/max amount/cost) into a Decimal.
    For max limits, None means no limit. For min limits, None means default (usually 0).
    """
    if limit_value_str is None or str(limit_value_str).strip() == "":
        return None if is_max_limit else default_if_none_or_invalid

    try:
        value = Decimal(str(limit_value_str))
        if value >= Decimal("0"):  # Allow zero for min, any non-negative for max
            return value
        logger_instance.warning(
            f"Invalid {limit_name} value '{limit_value_str}' for {symbol_for_log} "
            f"(must be non-negative). Using {'no limit' if is_max_limit else default_if_none_or_invalid}."
        )
    except InvalidOperation:
        logger_instance.warning(
            f"Cannot parse {limit_name} string '{limit_value_str}' for {symbol_for_log}. "
            f"Using {'no limit' if is_max_limit else default_if_none_or_invalid}."
        )
    
    return None if is_max_limit else default_if_none_or_invalid


def _estimate_trade_cost(
    size: Decimal,
    price: Decimal,
    contract_size_decimal: Decimal,
    use_linear_spot_logic: bool,
    use_inverse_logic: bool,
    symbol_for_log: str,
    logger_instance: logging.Logger,
) -> Optional[Decimal]:
    """Estimates the cost of a trade in quote currency."""
    if price <= Decimal("0"):
        logger_instance.error(
            f"Cost estimation failed for {symbol_for_log}: Price ({price}) must be positive."
        )
        return None
    if contract_size_decimal <= Decimal("0"): # Should be validated before this helper
        logger_instance.error(
            f"Cost estimation failed for {symbol_for_log}: Contract size ({contract_size_decimal}) must be positive."
        )
        return None

    if use_linear_spot_logic: # Spot or Linear futures
        return size * price * contract_size_decimal
    if use_inverse_logic: # Inverse futures
        return (size * contract_size_decimal) / price
    
    logger_instance.error(
        f"Cost estimation failed for {symbol_for_log}: Unknown market calculation type."
    )
    return None


def _calculate_size_for_target_cost(
    target_cost: Decimal,
    price: Decimal,
    contract_size_decimal: Decimal,
    use_linear_spot_logic: bool,
    use_inverse_logic: bool,
    symbol_for_log: str,
    logger_instance: logging.Logger,
) -> Optional[Decimal]:
    """Calculates the required position size to achieve a target cost."""
    if price <= Decimal("0"):
        logger_instance.error(
            f"Size-from-cost calculation failed for {symbol_for_log}: Price ({price}) must be positive."
        )
        return None
    if contract_size_decimal <= Decimal("0"): # Should be validated before this helper
        logger_instance.error(
            f"Size-from-cost calculation failed for {symbol_for_log}: "
            f"Contract size ({contract_size_decimal}) must be positive."
        )
        return None

    if use_linear_spot_logic:
        # Cost = Size * Price * ContractSize => Size = Cost / (Price * ContractSize)
        denominator = price * contract_size_decimal
        if denominator <= Decimal("0"):
            logger_instance.error(
                f"Size-from-cost calculation failed for {symbol_for_log} (linear/spot): "
                f"Denominator (price * contract_size = {denominator}) is not positive."
            )
            return None
        return target_cost / denominator
    if use_inverse_logic:
        # Cost = Size * ContractSize / Price => Size = (TargetCost * Price) / ContractSize
        return (target_cost * price) / contract_size_decimal
    
    logger_instance.error(
        f"Size-from-cost calculation failed for {symbol_for_log}: Unknown market calculation type."
    )
    return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: Union[float, Decimal, str],
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict[str, Any],
    exchange: ccxt.Exchange,
    logger: Optional[logging.Logger] = None,
) -> Optional[Decimal]:
    """
    Calculates the position size in base currency or contracts, applying exchange limits.

    Args:
        balance: Account balance in quote currency.
        risk_per_trade: Fraction of balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: Price at which to exit if the trade moves adversely.
        entry_price: Expected entry price of the trade.
        market_info: CCXT market information dictionary.
        exchange: CCXT exchange instance for precision formatting.
        logger: Optional logger instance.

    Returns:
        Calculated position size as Decimal, or None if calculation fails.
    """
    lg = logger or DEFAULT_LOGGER
    
    # --- Extract market properties ---
    symbol = market_info.get(KEY_SYMBOL, "UNKNOWN_SYMBOL")
    quote_currency = market_info.get(KEY_QUOTE_CURRENCY, "QUOTE")
    base_currency = market_info.get(KEY_BASE_CURRENCY, "BASE")
    is_contract_market = market_info.get(KEY_IS_CONTRACT, False)
    size_unit = "Contracts" if is_contract_market else base_currency

    # Determine calculation logic type (spot treated as linear)
    # If 'linear' key exists, it determines if a contract is linear.
    # If 'linear' key is absent, default for contracts is True (linear).
    # Spot markets are always 'linear-like'.
    use_linear_spot_logic = market_info.get(KEY_LINEAR, not is_contract_market) or not is_contract_market
    # Inverse contracts must be explicitly marked and be contracts.
    use_inverse_logic = market_info.get(KEY_INVERSE, False) and is_contract_market

    if use_linear_spot_logic and use_inverse_logic:
        lg.error(
            f"Position sizing failed ({symbol}): Market ambiguously defined as both linear-like and inverse. "
            f"Market Info: linear={market_info.get(KEY_LINEAR)}, inverse={market_info.get(KEY_INVERSE)}, "
            f"is_contract={is_contract_market}"
        )
        return None
    if not use_linear_spot_logic and not use_inverse_logic and is_contract_market:
        # This implies a contract type that is neither linear nor inverse, or misconfigured market_info
        lg.error(
            f"Position sizing failed ({symbol}): Unsupported contract type. Must be linear or inverse. "
            f"Market Info: linear={market_info.get(KEY_LINEAR)}, inverse={market_info.get(KEY_INVERSE)}, "
            f"is_contract={is_contract_market}, type={market_info.get(KEY_TYPE)}"
        )
        return None

    # --- Input Validation ---
    if not (isinstance(balance, Decimal) and balance > Decimal("0")):
        lg.error(f"Position sizing failed ({symbol}): Invalid or non-positive balance ({balance}).")
        return None

    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal("0") <= risk_decimal <= Decimal("1")):
            lg.error(
                f"Position sizing failed ({symbol}): risk_per_trade ({risk_decimal}) must be between 0 and 1."
            )
            return None
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(
            f"Position sizing failed ({symbol}): Invalid risk_per_trade ('{risk_per_trade}'). Error: {e}"
        )
        return None

    if not (isinstance(initial_stop_loss_price, Decimal) and initial_stop_loss_price > Decimal("0")):
        lg.error(
            f"Position sizing failed ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price})."
        )
        return None
    if not (isinstance(entry_price, Decimal) and entry_price > Decimal("0")):
        lg.error(f"Position sizing failed ({symbol}): Invalid entry_price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed ({symbol}): Stop loss price cannot be equal to entry price.")
        return None
    
    if KEY_LIMITS not in market_info or KEY_PRECISION_INFO not in market_info:
        lg.error(f"Position sizing failed ({symbol}): Market info missing '{KEY_LIMITS}' or '{KEY_PRECISION_INFO}'.")
        return None

    # --- Main Calculation Logic ---
    try:
        risk_amount_in_quote = balance * risk_decimal
        if risk_decimal > Decimal("0") and risk_amount_in_quote <= Decimal("0"):
            # This can happen if balance is extremely small, leading to risk_amount_in_quote rounding to 0 or less
            lg.warning(
                f"Calculated risk amount {risk_amount_in_quote:.8f} {quote_currency} for {symbol} "
                f"(risk {risk_decimal:.2%}, balance {balance:.4f}) is non-positive. "
                f"Cannot calculate position size if risk amount is not positive."
            )
            # If risk_per_trade is 0, risk_amount_in_quote will be 0, proceed if allowed by limits.
            # But if risk_per_trade > 0 and risk_amount_in_quote <= 0, this is an issue.
            if risk_decimal > Decimal("0"):
                return None


        stop_loss_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if stop_loss_distance_per_unit <= Decimal("0"): # Should be caught by SL != Entry, but good to double check
            lg.error(
                f"Position sizing failed ({symbol}): Stop loss distance is non-positive ({stop_loss_distance_per_unit})."
            )
            return None

        # Contract size: typically 1 for spot. For contracts, it's the value of one contract.
        contract_size_str = market_info.get(KEY_CONTRACT_SIZE, "1")
        try:
            contract_size_decimal = Decimal(str(contract_size_str))
            if contract_size_decimal <= Decimal("0"):
                lg.warning(
                    f"Invalid contract size '{contract_size_str}' for {symbol} (must be > 0). Defaulting to 1."
                )
                contract_size_decimal = Decimal("1")
        except (InvalidOperation, ValueError, TypeError):
            lg.warning(
                f"Error parsing contract size '{contract_size_str}' for {symbol}. Defaulting to 1."
            )
            contract_size_decimal = Decimal("1")
        
        # --- Initial position size calculation based on risk ---
        calculated_size: Optional[Decimal] = None
        if use_linear_spot_logic:
            # For spot or linear contracts: Size = RiskAmount / (SLDistance * ContractSize)
            denominator = stop_loss_distance_per_unit * contract_size_decimal
            if denominator > Decimal("0"):
                calculated_size = risk_amount_in_quote / denominator
            else:
                lg.error(
                    f"Position sizing failed ({symbol}): Denominator for linear/spot size calc is not positive "
                    f"(SLDist={stop_loss_distance_per_unit}, ContrSize={contract_size_decimal})."
                )
                return None
        elif use_inverse_logic:
            # For inverse contracts: Size = RiskAmount / (ContractSize * |1/Entry - 1/SL|)
            if entry_price == Decimal("0") or initial_stop_loss_price == Decimal("0"): # Redundant due to >0 checks, but safe
                lg.error(
                    f"Position sizing failed ({symbol}): Entry or SL price is zero for inverse contract calculation."
                )
                return None
            loss_per_contract_in_quote = contract_size_decimal * abs(
                (Decimal("1") / entry_price) - (Decimal("1") / initial_stop_loss_price)
            )
            if loss_per_contract_in_quote > Decimal("0"):
                calculated_size = risk_amount_in_quote / loss_per_contract_in_quote
            else:
                lg.error(
                    f"Position sizing failed ({symbol}): Loss per contract for inverse calc is not positive "
                    f"({loss_per_contract_in_quote})."
                )
                return None
        # No `else` needed due to earlier checks on use_linear_spot_logic and use_inverse_logic

        if not (calculated_size and calculated_size > Decimal("0")):
            # This can happen if risk_amount_in_quote is zero (e.g. risk_per_trade is 0%)
            # If risk_per_trade is 0%, calculated_size will be 0. This might be intentional.
            # However, if risk_per_trade > 0% and calculated_size is still <=0, it's an issue.
            if risk_decimal > Decimal("0"):
                lg.error(
                    f"Initial position size calculation resulted in zero or negative size: {calculated_size}. "
                    f"RiskAmt={risk_amount_in_quote:.4f} {quote_currency}, SLDist={stop_loss_distance_per_unit}, "
                    f"ContrSize={contract_size_decimal}. Check inputs and market data for {symbol}."
                )
                return None
            elif calculated_size is None: # Should have been caught by denominator checks
                lg.error(f"Initial position size calculation failed for {symbol}, result is None.")
                return None
            # If calculated_size is 0 due to 0 risk, proceed. Limits might still apply.
            if calculated_size < Decimal("0"): # Strictly negative size is always an error
                 lg.error(f"Initial position size calculation resulted in negative size: {calculated_size} for {symbol}.")
                 return None


        price_precision_log_fmt = get_price_precision(market_info, lg)
        lg.info(
            f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_decimal:.2%}, "
            f"RiskAmount={risk_amount_in_quote:.4f} {quote_currency}"
        )
        lg.info(
            f"  Entry={entry_price:.{price_precision_log_fmt}f}, SL={initial_stop_loss_price:.{price_precision_log_fmt}f}, "
            f"SLDistPerUnit={stop_loss_distance_per_unit:.{price_precision_log_fmt}f}"
        )
        lg.info(
            f"  ContractSize={contract_size_decimal}, Initial Calculated Size = {calculated_size:.8f} {size_unit}"
        )

        # --- Parse Market Limits ---
        limits_dict = market_info.get(KEY_LIMITS, {})
        if not isinstance(limits_dict, dict):
            lg.warning(f"Market limits for {symbol} is not a dictionary. Using default limits.")
            limits_dict = {}

        amount_limits_dict = limits_dict.get(KEY_AMOUNT, {})
        if not isinstance(amount_limits_dict, dict): amount_limits_dict = {}
        cost_limits_dict = limits_dict.get(KEY_COST, {})
        if not isinstance(cost_limits_dict, dict): cost_limits_dict = {}
        
        min_amount = _parse_decimal_limit(
            amount_limits_dict.get(KEY_MIN), "min amount", symbol, lg, Decimal("0")
        )
        max_amount = _parse_decimal_limit(
            amount_limits_dict.get(KEY_MAX), "max amount", symbol, lg, Decimal("0"), is_max_limit=True
        ) # max_amount can be None
        min_cost = _parse_decimal_limit(
            cost_limits_dict.get(KEY_MIN), "min cost", symbol, lg, Decimal("0")
        )
        max_cost = _parse_decimal_limit(
            cost_limits_dict.get(KEY_MAX), "max cost", symbol, lg, Decimal("0"), is_max_limit=True
        ) # max_cost can be None


        # --- Adjust size based on amount limits ---
        adjusted_size = calculated_size
        if adjusted_size < min_amount: # min_amount is Decimal("0") if not specified or invalid
            lg.warning(
                f"Calculated size {calculated_size:.8f} {size_unit} is less than min amount limit {min_amount} {size_unit} "
                f"for {symbol}. Adjusting size up to min amount."
            )
            adjusted_size = min_amount
        if max_amount is not None and adjusted_size > max_amount:
            lg.warning(
                f"Calculated size {calculated_size:.8f} {size_unit} is greater than max amount limit {max_amount} {size_unit} "
                f"for {symbol}. Adjusting size down to max amount."
            )
            adjusted_size = max_amount
        
        if adjusted_size != calculated_size:
            lg.info(f"  Size after amount limits: {adjusted_size:.8f} {size_unit}")

        # --- Adjust size based on cost limits ---
        # Cost estimation price should be entry_price for consistency
        estimated_cost = _estimate_trade_cost(
            adjusted_size, entry_price, contract_size_decimal,
            use_linear_spot_logic, use_inverse_logic, symbol, lg
        )
        if estimated_cost is None:
            lg.error(f"Failed to estimate cost for {symbol} with size {adjusted_size:.8f}. Aborting.")
            return None
        lg.debug(f"  Estimated cost for size {adjusted_size:.8f} {size_unit}: {estimated_cost:.4f} {quote_currency}")

        # Min Cost Adjustment
        if estimated_cost < min_cost: # min_cost is Decimal("0") if not specified
            lg.warning(
                f"Estimated cost {estimated_cost:.4f} {quote_currency} is less than min cost limit {min_cost} {quote_currency} "
                f"for {symbol}. Attempting to increase size to meet min cost."
            )
            required_size_for_min_cost = _calculate_size_for_target_cost(
                min_cost, entry_price, contract_size_decimal,
                use_linear_spot_logic, use_inverse_logic, symbol, lg
            )
            if required_size_for_min_cost is None or required_size_for_min_cost <= Decimal("0"):
                lg.error(
                    f"Could not calculate a valid size to meet min_cost {min_cost} {quote_currency} for {symbol}. Aborting."
                )
                return None
            
            lg.info(f"  Required size to meet min_cost {min_cost} {quote_currency}: {required_size_for_min_cost:.8f} {size_unit}")

            # Check new size against amount limits
            if required_size_for_min_cost < min_amount:
                lg.error(
                    f"Size required for min_cost ({required_size_for_min_cost:.8f} {size_unit}) is less than "
                    f"min_amount ({min_amount} {size_unit}) for {symbol}. Limits conflict. Aborting."
                )
                return None
            if max_amount is not None and required_size_for_min_cost > max_amount:
                lg.error(
                    f"Size required for min_cost ({required_size_for_min_cost:.8f} {size_unit}) exceeds "
                    f"max_amount ({max_amount} {size_unit}) for {symbol}. Cannot meet min_cost. Aborting."
                )
                return None
            
            adjusted_size = required_size_for_min_cost
            lg.info(f"  Adjusted size to meet min_cost: {adjusted_size:.8f} {size_unit}")
            
            # Re-estimate cost for max_cost check
            estimated_cost = _estimate_trade_cost(
                adjusted_size, entry_price, contract_size_decimal,
                use_linear_spot_logic, use_inverse_logic, symbol, lg
            )
            if estimated_cost is None: # Should not happen if previous steps were successful
                lg.error(f"Failed to re-estimate cost for {symbol} after min_cost adjustment. Aborting.")
                return None
            lg.debug(f"  Re-estimated cost after min_cost adj: {estimated_cost:.4f} {quote_currency}")


        # Max Cost Adjustment
        if max_cost is not None and estimated_cost > max_cost:
            lg.warning(
                f"Estimated cost {estimated_cost:.4f} {quote_currency} exceeds max cost limit {max_cost} {quote_currency} "
                f"for {symbol}. Reducing size to meet max cost."
            )
            size_for_max_cost = _calculate_size_for_target_cost(
                max_cost, entry_price, contract_size_decimal,
                use_linear_spot_logic, use_inverse_logic, symbol, lg
            )
            if size_for_max_cost is None or size_for_max_cost <= Decimal("0"):
                lg.error(
                    f"Could not calculate a valid size for max_cost {max_cost} {quote_currency} for {symbol}. Aborting."
                )
                return None

            lg.info(f"  Calculated size to meet max_cost {max_cost} {quote_currency}: {size_for_max_cost:.8f} {size_unit}")

            # Check new size against min_amount (it's implicitly <= max_amount)
            if size_for_max_cost < min_amount:
                lg.error(
                    f"Size reduced for max_cost ({size_for_max_cost:.8f} {size_unit}) is less than "
                    f"min_amount ({min_amount} {size_unit}) for {symbol}. Limits conflict. Aborting."
                )
                return None
            
            adjusted_size = size_for_max_cost
            lg.info(f"  Adjusted size to meet max_cost: {adjusted_size:.8f} {size_unit}")

        # --- Apply exchange precision to amount ---
        final_size: Optional[Decimal] = None
        try:
            # CCXT's amount_to_precision expects symbol and a number (float or string usually)
            final_size_str = exchange.amount_to_precision(symbol, float(adjusted_size))
            final_size = Decimal(final_size_str)
            lg.info(
                f"Applied exchange amount precision (via CCXT): {adjusted_size:.8f} -> {final_size} {size_unit}"
            )
        except Exception as e_ccxt_prec:
            lg.warning(
                f"Error applying CCXT amount_to_precision for {symbol} on size {adjusted_size:.8f}: {e_ccxt_prec}. "
                "Attempting manual quantization using market_info['amountPrecision']."
            )
            # Fallback: use 'amountPrecision' if available (assumed to be number of decimal places)
            # Or use CCXT's market['precision']['amount'] (which is a step size like 0.001)
            amount_precision_step = None
            precision_data = market_info.get(KEY_PRECISION_INFO, {})
            if isinstance(precision_data, dict) and KEY_AMOUNT in precision_data:
                amount_precision_step = Decimal(str(precision_data[KEY_AMOUNT])) # e.g., Decimal('0.001')
            
            if amount_precision_step and amount_precision_step > Decimal("0"):
                final_size = (adjusted_size // amount_precision_step) * amount_precision_step # Floor to step size
                lg.info(
                    f"Applied manual amount quantization (floor to step {amount_precision_step}): "
                    f"{adjusted_size:.8f} -> {final_size} {size_unit}"
                )
            else: # Fallback to custom 'amountPrecision' key or default
                num_decimal_places = market_info.get(KEY_AMOUNT_PRECISION_FALLBACK, 8) # Default 8 decimal places
                try:
                    num_decimal_places = int(num_decimal_places)
                    quant_factor = Decimal('1e-' + str(num_decimal_places))
                    final_size = adjusted_size.quantize(quant_factor, rounding=ROUND_DOWN)
                    lg.info(
                        f"Applied manual amount quantization (ROUND_DOWN to {num_decimal_places} places): "
                        f"{adjusted_size:.8f} -> {final_size} {size_unit}"
                    )
                except (ValueError, TypeError):
                    lg.error(
                        f"Invalid {KEY_AMOUNT_PRECISION_FALLBACK} ('{num_decimal_places}') for {symbol}. "
                        f"Cannot apply precision. Using unrounded size: {adjusted_size:.8f}"
                    )
                    final_size = adjusted_size # As a last resort, use the unrounded size

        if not (final_size is not None and final_size > Decimal("0")): # Must be strictly positive unless min_amount is 0
            if final_size == Decimal("0") and min_amount == Decimal("0"):
                lg.warning(f"Final position size is 0 for {symbol}, which is allowed as min_amount is 0.")
            else:
                lg.error(
                    f"Final position size is zero or negative ({final_size}) after precision for {symbol}. Aborting."
                )
                return None
        
        if final_size < min_amount:
            lg.error(
                f"Final size {final_size} {size_unit} is less than min_amount {min_amount} {size_unit} "
                f"after precision adjustment for {symbol}. Aborting."
            )
            return None

        # --- Final cost check after precision ---
        # Precision adjustment might change cost, potentially violating min_cost.
        final_estimated_cost = _estimate_trade_cost(
            final_size, entry_price, contract_size_decimal,
            use_linear_spot_logic, use_inverse_logic, symbol, lg
        )
        if final_estimated_cost is None:
            lg.error(f"Failed to estimate final cost for {symbol} with final size {final_size}. Aborting.")
            return None

        if final_estimated_cost < min_cost:
            # If min_cost is 0, this condition is fine. Only an issue if min_cost > 0.
            if min_cost > Decimal("0"):
                lg.error(
                    f"Final estimated cost {final_estimated_cost:.4f} {quote_currency} for final size {final_size} {size_unit} "
                    f"is less than min_cost {min_cost} {quote_currency} for {symbol} (likely due to precision rounding). Aborting."
                )
                return None
        
        # Max cost should ideally not be violated by rounding down size, but good to be aware.
        if max_cost is not None and final_estimated_cost > max_cost:
            lg.warning( # Warning because size was rounded down, so cost should decrease or stay same
                f"Final estimated cost {final_estimated_cost:.4f} {quote_currency} for final size {final_size} {size_unit} "
                f"still exceeds max_cost {max_cost} {quote_currency} for {symbol}. This is unexpected after precision."
            )
            # Depending on policy, might return None here. For now, proceed with warning.


        lg.info(
            f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}"
            f" (Est. Cost: {final_estimated_cost:.4f} {quote_currency})"
        )
        return final_size

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Decimal/Type error during position size calculation for {symbol}: {e}", exc_info=False)
    except Exception as e:
        lg.error(f"Unexpected error calculating position size for {symbol}: {e}", exc_info=True)
    
    return None
```