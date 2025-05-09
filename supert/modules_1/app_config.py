# File: app_config.py
"""
Defines the Config class for loading, validating, and storing application
configuration parameters from environment variables. Initializes a global CONFIG instance.
"""
import os
import traceback
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any

try:
    from dotenv import load_dotenv
    from colorama import Fore, Back, Style
except ImportError:
    # Fallback for imports, main script should handle critical import errors
    class Fore: pass
    class Back: pass
    class Style: pass
    for name in ['RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE', 'RESET', 'DIM', 'BRIGHT', 'NORMAL']:
        setattr(Fore, name, '')
        setattr(Back, name, '')
        setattr(Style, name, '')
    def load_dotenv(): pass


from logger_config import logger

load_dotenv()  # Load secrets from the hidden .env scroll (if present)
getcontext().prec = 18  # Set Decimal precision for financial exactitude

class Config:
    """Loads, validates, and stores configuration parameters from environment variables.
    Provides robust type casting, default value handling, validation, and logging.
    """

    def __init__(self) -> None:
        """Initializes configuration by loading and validating environment variables."""
        logger.info(f"{Fore.MAGENTA}--- Summoning Configuration Runes ---{Style.RESET_ALL}")
        # --- API Credentials - Keys to the Exchange Vault ---
        self.api_key: str | None = self._get_env("BYBIT_API_KEY", required=True, color=Fore.RED, secret=True)
        self.api_secret: str | None = self._get_env("BYBIT_API_SECRET", required=True, color=Fore.RED, secret=True)

        # --- Trading Parameters - Core Incantation Variables ---
        self.symbol: str = self._get_env(
            "SYMBOL", "BTC/USDT:USDT", color=Fore.YELLOW
        )
        self.interval: str = self._get_env(
            "INTERVAL", "1m", color=Fore.YELLOW
        )
        self.leverage: int = self._get_env(
            "LEVERAGE", 10, cast_type=int, color=Fore.YELLOW
        )
        self.sleep_seconds: int = self._get_env(
            "SLEEP_SECONDS", 10, cast_type=int, color=Fore.YELLOW
        )

        # --- Strategy Selection - Choosing the Path of Magic ---
        self.strategy_name: str = self._get_env("STRATEGY_NAME", "DUAL_SUPERTREND", color=Fore.CYAN).upper()
        self.valid_strategies: list[str] = ["DUAL_SUPERTREND", "STOCHRSI_MOMENTUM", "EHLERS_FISHER", "EMA_CROSS"]
        if self.strategy_name not in self.valid_strategies:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}Invalid STRATEGY_NAME '{self.strategy_name}'. Valid paths: {self.valid_strategies}{Style.RESET_ALL}"
            )
            raise ValueError(
                f"Invalid STRATEGY_NAME '{self.strategy_name}'. Valid options are: {self.valid_strategies}"
            )
        logger.info(f"{Fore.CYAN}Chosen Strategy Path: {self.strategy_name}{Style.RESET_ALL}")

        # --- Risk Management - Wards Against Ruin ---
        self.risk_per_trade_percentage: Decimal = self._get_env(
            "RISK_PER_TRADE_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )
        self.atr_stop_loss_multiplier: Decimal = self._get_env(
            "ATR_STOP_LOSS_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN
        )
        self.atr_take_profit_multiplier: Decimal = self._get_env(
            "ATR_TAKE_PROFIT_MULTIPLIER", "1.5", cast_type=Decimal, color=Fore.GREEN
        )
        self.max_order_usdt_amount: Decimal = self._get_env(
            "MAX_ORDER_USDT_AMOUNT", "500.0", cast_type=Decimal, color=Fore.GREEN
        )
        self.required_margin_buffer: Decimal = self._get_env(
            "REQUIRED_MARGIN_BUFFER", "1.05", cast_type=Decimal, color=Fore.GREEN
        )

        # --- Native Stop-Loss & Trailing Stop-Loss (Exchange Native - Bybit V5) ---
        self.trailing_stop_percentage: Decimal = self._get_env(
            "TRAILING_STOP_PERCENTAGE", "0.005", cast_type=Decimal, color=Fore.GREEN
        )
        self.trailing_stop_activation_offset_percent: Decimal = self._get_env(
            "TRAILING_STOP_ACTIVATION_PRICE_OFFSET_PERCENT", "0.001", cast_type=Decimal, color=Fore.GREEN
        )

        # --- Strategy-Specific Parameters ---
        self.st_atr_length: int = self._get_env(
            "ST_ATR_LENGTH", 7, cast_type=int, color=Fore.CYAN
        )
        self.st_multiplier: Decimal = self._get_env(
            "ST_MULTIPLIER", "2.5", cast_type=Decimal, color=Fore.CYAN
        )
        self.confirm_st_atr_length: int = self._get_env(
            "CONFIRM_ST_ATR_LENGTH", 5, cast_type=int, color=Fore.CYAN
        )
        self.confirm_st_multiplier: Decimal = self._get_env(
            "CONFIRM_ST_MULTIPLIER", "2.0", cast_type=Decimal, color=Fore.CYAN
        )
        self.stochrsi_rsi_length: int = self._get_env(
            "STOCHRSI_RSI_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_stoch_length: int = self._get_env(
            "STOCHRSI_STOCH_LENGTH", 14, cast_type=int, color=Fore.CYAN
        )
        self.stochrsi_k_period: int = self._get_env(
            "STOCHRSI_K_PERIOD", 3, cast_type=int, color=Fore.CYAN
