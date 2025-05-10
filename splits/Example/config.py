# File: config.py
# -*- coding: utf-8 -*-

"""
Configuration Class Definition
"""

import os
from decimal import Decimal
from typing import Optional, Literal


class Config:
    """
    Configuration class holding settings for the Bybit helper modules.
    Adapt attribute values as needed or load from environment/files.
    """

    # Retry mechanism settings (used by external decorator)
    RETRY_COUNT: int = 3
    RETRY_DELAY_SECONDS: float = 2.0

    # Position / Order settings
    POSITION_QTY_EPSILON: Decimal = Decimal(
        "1e-9"
    )  # Threshold for treating qty as zero
    QUOTE_PRECISION: int = 2  # Assumed precision for quote currency (e.g., USDT)
    DEFAULT_SLIPPAGE_PCT: Decimal = Decimal(
        "0.005"
    )  # Default max slippage for market orders
    ORDER_BOOK_FETCH_LIMIT: int = 25  # Default depth for fetch_l2_order_book
    SHALLOW_OB_FETCH_DEPTH: int = 5  # Depth used for slippage check analysis

    # Symbol / Market settings
    SYMBOL: str = "DOT/USDT:USDT"  # Default symbol
    USDT_SYMBOL: str = "USDT"  # Quote currency symbol for balance checks
    EXPECTED_MARKET_TYPE: Literal["swap", "future", "spot", "option"] = "swap"
    EXPECTED_MARKET_LOGIC: Optional[Literal["linear", "inverse"]] = "linear"

    # Exchange connection settings
    EXCHANGE_ID: str = "bybit"
    API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")  # Load from environment
    API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")  # Load from environment
    DEFAULT_RECV_WINDOW: int = 10000
    TESTNET_MODE: bool = False  # Set to False for production

    # Account settings
    DEFAULT_LEVERAGE: int = 10
    DEFAULT_MARGIN_MODE: Literal["cross", "isolated"] = "cross"
    DEFAULT_POSITION_MODE: Literal["one-way", "hedge"] = "one-way"

    # Fees (Example Bybit VIP 0 - Update based on your actual fees)
    TAKER_FEE_RATE: Decimal = Decimal("0.00055")
    MAKER_FEE_RATE: Decimal = Decimal("0.0002")

    # SMS Alerts (Optional)
    ENABLE_SMS_ALERTS: bool = False
    SMS_RECIPIENT_NUMBER: Optional[str] = None
    SMS_TIMEOUT_SECONDS: int = 30

    # Side / Position Constants
    SIDE_BUY: str = "buy"
    SIDE_SELL: str = "sell"
    POS_LONG: str = "LONG"
    POS_SHORT: str = "SHORT"
    POS_NONE: str = "NONE"


# --- END OF FILE config.py ---

# ---------------------------------------------------------------------------
