# File: risk_calculator.py
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

# Third-party Libraries
try:
    import ccxt
    from colorama import Fore, Style
except ImportError:
    class DummyCCXTExchange: pass
    class DummyCCXT:
        Exchange = DummyCCXTExchange
    ccxt = DummyCCXT() # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Style = DummyColor(), DummyColor()

# Custom module imports
from logger_setup import logger
from config import CONFIG
from utils import format_amount


def calculate_position_size(
    exchange: ccxt.Exchange, # Added exchange for format_amount
    symbol: str,             # Added symbol for format_amount
    equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int,
) -> tuple[Decimal | None, Decimal | None]:
    """Calculates position size and estimated margin based on risk, using Decimal."""
    logger.debug(
        f"Risk Calc: Equity={equity:.4f}, Risk%={risk_per_trade_pct:.4%}, Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x"
    )
    if not (entry_price > 0 and stop_loss_price > 0):
        logger.error(f"{Fore.RED}Risk Calc: Invalid entry/SL price.{Style.RESET_ALL}")
        return None, None
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff < CONFIG.position_qty_epsilon: # Use config epsilon
        logger.error(f"{Fore.RED}Risk Calc: Entry/SL prices too close ({price_diff:.8f}).{Style.RESET_ALL}")
        return None, None
    if not 0 < risk_per_trade_pct < 1:
        logger.error(f"{Fore.RED}Risk Calc: Invalid risk %: {risk_per_trade_pct:.4%}.{Style.RESET_ALL}")
        return None, None
    if equity <= CONFIG.position_qty_epsilon: # Use config epsilon
        logger.error(f"{Fore.RED}Risk Calc: Invalid equity: {equity:.4f}.{Style.RESET_ALL}")
        return None, None
    if leverage <= 0:
        logger.error(f"{Fore.RED}Risk Calc: Invalid leverage: {leverage}{Style.RESET_ALL}")
        return None, None

    risk_amount_usdt = equity * risk_per_trade_pct
    quantity_raw = risk_amount_usdt / price_diff

    try:
        quantity_precise_str = format_amount(exchange, symbol, quantity_raw)
        quantity_precise = Decimal(quantity_precise_str)
    except Exception as e:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Failed precision formatting for quantity {quantity_raw:.8f}. Using raw. Error: {e}{Style.RESET_ALL}"
        )
        # Ensure fallback quantization has enough precision
        quantity_precise = quantity_raw.quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)


    if quantity_precise <= CONFIG.position_qty_epsilon:
        logger.warning(
            f"{Fore.YELLOW}Risk Calc: Calculated quantity negligible ({quantity_precise:.8f}). RiskAmt={risk_amount_usdt:.4f}, PriceDiff={price_diff:.4f}{Style.RESET_ALL}"
        )
        return None, None

    pos_value_usdt = quantity_precise * entry_price
    required_margin = pos_value_usdt / Decimal(leverage)
    logger.debug(
        f"Risk Calc Result: Qty={quantity_precise:.8f}, EstValue={pos_value_usdt:.4f}, EstMargin={required_margin:.4f}"
    )
    return quantity_precise, required_margin

# End of risk_calculator.py
```

```python
