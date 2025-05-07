# File: risk_manager.py
import logging
import ccxt
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any, Dict, Optional

# Import constants and utilities
import constants
# utils import might not be needed if precision helpers are passed or handled differently

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Needed for formatting
    logger: Optional[logging.Logger] = None,
    quote_currency: str = 'USDT' # Pass quote currency
) -> Optional[Decimal]:
    """
    Calculates the position size based on risk, SL distance, balance, and market constraints.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or not isinstance(balance, Decimal) or balance <= 0: lg.error(f"Position sizing failed ({symbol}): Invalid or zero balance ({balance})."); return None
    if not isinstance(risk_per_trade, (float, int)) or not (0 < risk_per_trade < 1): lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1."); return None
    if initial_stop_loss_price is None or not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= 0: lg.error(f"Position sizing failed ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price})."); return None
    if entry_price is None or not isinstance(entry_price, Decimal) or entry_price <= 0: lg.error(f"Position sizing failed ({symbol}): Invalid entry_price ({entry_price})."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Position sizing failed ({symbol}): Stop loss price cannot be equal to entry price."); return None
    if 'limits' not in market_info or 'precision' not in market_info: lg.error(f"Position sizing failed ({symbol}): Market info missing 'limits' or 'precision'."); return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0: lg.error(f"Position sizing failed ({symbol}): Stop loss distance is zero or negative ({sl_distance_per_unit})."); return None

        contract_size_str = market_info.get('contractSize', '1')
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0: raise ValueError("Contract size must be positive")
        except (InvalidOperation, ValueError, TypeError): lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}, using 1."); contract_size = Decimal('1')

        calculated_size: Optional[Decimal] = None
        if market_info.get('linear', True) or not is_contract:
             denominator = sl_distance_per_unit * contract_size
             if denominator > 0: calculated_size = risk_amount_quote / denominator
             else: lg.error(f"Position sizing failed ({symbol}): Denominator zero/negative (SL Dist: {sl_distance_per_unit}, ContractSize: {contract_size})."); return None
        else: # Inverse Contract Placeholder
             lg.error(f"{constants.NEON_RED}Inverse contract sizing not fully implemented. Aborting sizing for {symbol}.{constants.RESET}"); return None

        if calculated_size is None or calculated_size <= 0: lg.error(f"Initial position size calculation resulted in zero or negative: {calculated_size}. RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_per_unit}, ContractSize={contract_size}"); return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  ContractSize={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount')

        min_amount_str = amount_limits.get('min'); max_amount_str = amount_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')
        min_cost_str = cost_limits.get('min'); max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')

        adjusted_size = calculated_size
        if adjusted_size < min_amount: lg.warning(f"{constants.NEON_YELLOW}Calculated size {calculated_size:.8f} is below min amount {min_amount}. Adjusting to min amount.{constants.RESET}"); adjusted_size = min_amount
        elif adjusted_size > max_amount: lg.warning(f"{constants.NEON_YELLOW}Calculated size {calculated_size:.8f} exceeds max amount {max_amount}. Adjusting to max amount.{constants.RESET}"); adjusted_size = max_amount

        estimated_cost: Optional[Decimal] = None
        if market_info.get('linear', True) or not is_contract: estimated_cost = adjusted_size * entry_price * contract_size
        else:
             if entry_price > 0: estimated_cost = adjusted_size * contract_size / entry_price
             else: lg.error(f"Cannot estimate cost for inverse contract {symbol}: Entry price is zero."); return None
        if estimated_cost is None: return None
        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Estimated Cost={estimated_cost:.4f} {quote_currency}")

        if min_cost > 0 and estimated_cost < min_cost :
             lg.warning(f"{constants.NEON_YELLOW}Estimated cost {estimated_cost:.4f} is below min cost {min_cost}. Attempting to increase size.{constants.RESET}")
             required_size_for_min_cost: Optional[Decimal] = None; denominator = Decimal('0')
             if market_info.get('linear', True) or not is_contract:
                 denominator = entry_price * contract_size
                 if denominator > 0: required_size_for_min_cost = min_cost / denominator
             else:
                 denominator = contract_size
                 if denominator > 0 and entry_price > 0: required_size_for_min_cost = min_cost * entry_price / contract_size
             if required_size_for_min_cost is None or denominator <= 0: lg.error("Cannot calculate required size for min cost: Denominator zero/negative or invalid price."); return None
             lg.info(f"  Required size to meet min cost: {required_size_for_min_cost:.8f} {size_unit}")
             if required_size_for_min_cost > max_amount: lg.error(f"{constants.NEON_RED}Cannot meet min cost {min_cost} without exceeding max amount limit {max_amount}. Aborted.{constants.RESET}"); return None
             if required_size_for_min_cost < min_amount: lg.error(f"{constants.NEON_RED}Cannot meet min cost: Required size {required_size_for_min_cost:.8f} is below min amount {min_amount}. Aborted.{constants.RESET}"); return None
             else:
                 lg.info(f"  Adjusting size to meet min cost: {adjusted_size:.8f} -> {required_size_for_min_cost:.8f}"); adjusted_size = required_size_for_min_cost
                 if market_info.get('linear', True) or not is_contract: estimated_cost = adjusted_size * entry_price * contract_size
                 else: estimated_cost = adjusted_size * contract_size / entry_price if entry_price > 0 else Decimal('inf')

        elif max_cost > 0 and estimated_cost > max_cost:
             lg.warning(f"{constants.NEON_YELLOW}Estimated cost {estimated_cost:.4f} exceeds max cost {max_cost}. Reducing size.{constants.RESET}")
             adjusted_size_for_max_cost: Optional[Decimal] = None; denominator = Decimal('0')
             if market_info.get('linear', True) or not is_contract:
                  denominator = entry_price * contract_size
                  if denominator > 0: adjusted_size_for_max_cost = max_cost / denominator
             else:
                  denominator = contract_size
                  if denominator > 0 and entry_price > 0: adjusted_size_for_max_cost = max_cost * entry_price / contract_size
             if adjusted_size_for_max_cost is None or denominator <= 0: lg.error("Cannot calculate max size for max cost: Denominator zero/negative or invalid price."); return None
             lg.info(f"  Reduced size allowed by max cost: {adjusted_size_for_max_cost:.8f} {size_unit}")
             if adjusted_size_for_max_cost < min_amount: lg.error(f"{constants.NEON_RED}Size reduced for max cost ({adjusted_size_for_max_cost:.8f}) is below min amount {min_amount}. Aborted.{constants.RESET}"); return None
             else: lg.info(f"  Adjusting size to meet max cost: {adjusted_size:.8f} -> {adjusted_size_for_max_cost:.8f}"); adjusted_size = adjusted_size_for_max_cost

        final_size: Optional[Decimal] = None; formatted_size_str: Optional[str] = None
        try:
            amount_str = f"{adjusted_size:.{getcontext().prec}f}"
            formatted_size_str = exchange.amount_to_precision(symbol, amount_str)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step size: {adjusted_size:.8f} -> {final_size} {size_unit}")
        except ccxt.ExchangeError as fmt_err:
             lg.warning(f"{constants.NEON_YELLOW}CCXT formatting error applying amount precision ({fmt_err}). Check market data or try manual rounding.{constants.RESET}")
             if isinstance(amount_precision_val, (float, str)):
                 try:
                     amount_step = Decimal(str(amount_precision_val))
                     if amount_step > 0: final_size = (adjusted_size // amount_step) * amount_step; lg.info(f"Applied manual amount step size ({amount_step}): {adjusted_size:.8f} -> {final_size} {size_unit}")
                     else: raise ValueError("Amount step size is not positive")
                 except (InvalidOperation, ValueError, TypeError) as manual_err: lg.error(f"Manual step size rounding failed: {manual_err}. Using unrounded size: {adjusted_size}", exc_info=True); final_size = adjusted_size
             else: lg.error(f"Cannot determine amount step size for manual rounding. Using unrounded size: {adjusted_size}"); final_size = adjusted_size
        except (InvalidOperation, ValueError, TypeError) as dec_err: lg.error(f"Error converting formatted size '{formatted_size_str}' back to Decimal: {dec_err}"); return None

        if final_size is None or final_size <= 0: lg.error(f"{constants.NEON_RED}Position size became zero or negative ({final_size}) after adjustments. Aborted.{constants.RESET}"); return None
        if final_size < min_amount: lg.error(f"{constants.NEON_RED}Final size {final_size} is below minimum amount {min_amount} after precision formatting. Aborted.{constants.RESET}"); return None

        final_cost: Optional[Decimal] = None
        if market_info.get('linear', True) or not is_contract: final_cost = final_size * entry_price * contract_size
        else: final_cost = final_size * contract_size / entry_price if entry_price > 0 else None
        if final_cost is not None and min_cost > 0 and final_cost < min_cost: lg.error(f"{constants.NEON_RED}Final size {final_size} results in cost {final_cost:.4f} which is below minimum cost {min_cost}. Exchange limits conflict? Aborted.{constants.RESET}"); return None

        lg.info(f"{constants.NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{constants.RESET}")
        return final_size

    except (InvalidOperation, ValueError, TypeError) as e: lg.error(f"{constants.NEON_RED}Error during position size calculation ({symbol}) (Decimal/Type Error): {e}{constants.RESET}", exc_info=False); return None
    except Exception as e: lg.error(f"{constants.NEON_RED}Unexpected error calculating position size for {symbol}: {e}{constants.RESET}", exc_info=True); return None

```

```python
