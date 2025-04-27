```python
# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Termux Trading Spell (v2.1 - Precision & Robustness Enhanced)
# Conjures market insights and executes trades on Bybit Futures with refined precision and improved error handling.

import os
import time
import logging
import sys
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation

# Attempt to import necessary enchantments (dependencies)
try:
    import ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import init, Fore, Style, Back
    import requests # Often used by ccxt, good to check explicitly
except ImportError as e:
    # Provide specific guidance for Termux users or general Python environments
    init(autoreset=True) # Initialize colorama for potential error messages
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {missing_pkg}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Alternatively, to ensure all scrolls are present, cast:")
    print(f"{Style.BRIGHT}pip install ccxt python-dotenv pandas numpy tabulate colorama requests{Style.RESET_ALL}")
    sys.exit(f"Missing dependency: {missing_pkg}")

# Weave the Colorama magic into the terminal for colorful output
init(autoreset=True)

# Set Decimal precision if needed (higher precision uses more memory/CPU)
# Standard float precision is often sufficient for TA, but Decimal ensures accuracy for critical financial math.
# Default precision (28) is usually adequate unless dealing with extreme values.
# getcontext().prec = 30 # Example: Increase precision if needed

# --- Arcane Configuration ---
print(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.1...")

# Summon secrets and settings from the .env scroll
load_dotenv()

# Configure the Ethereal Log Scribe (Logger)
# Use a format that includes level name for better filtering and readability
log_formatter = logging.Formatter(
    fmt=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} {Style.BRIGHT}[%(levelname)s]{Style.RESET_ALL} {Fore.WHITE}%(message)s{Style.RESET_ALL}",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Pyrmethus")
logger.setLevel(logging.INFO) # Set to DEBUG for maximum verbosity, INFO for standard operation

# Avoid adding multiple handlers if the script is reloaded or run in certain environments
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

# Prevent log messages from propagating to the root logger if it has handlers
logger.propagate = False

class TradingConfig:
    """
    Holds the sacred parameters of our spell, enhanced with precision awareness
    and robust type casting from environment variables.
    """
    def __init__(self):
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW) # CCXT Unified Symbol (e.g., BTC/USDT:USDT)
        self.market_type: str = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW) # 'linear' (USDT margined) or 'inverse' (Coin margined)
        self.interval: str = self._get_env("INTERVAL", "1m", Fore.YELLOW) # e.g., '1m', '5m', '1h', '1d'

        # Financial Parameters (using Decimal for precision)
        self.risk_percentage: Decimal = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal) # e.g., 0.01 for 1%
        self.sl_atr_multiplier: Decimal = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal)
        self.tsl_activation_atr_multiplier: Decimal = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal)
        # Trailing stop distance as a percentage of price (e.g., 0.5 for 0.5%)
        self.trailing_stop_percent: Decimal = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal)
        self.position_qty_epsilon: Decimal = self._get_env("POSITION_QTY_EPSILON", "0.000001", Fore.YELLOW, cast_type=Decimal) # Threshold for treating position as closed

        # Order Execution Parameters
        self.sl_trigger_by: str = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW) # Options: LastPrice, MarkPrice, IndexPrice (check Bybit docs)
        self.tsl_trigger_by: str = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW) # Usually same as SL

        # API Credentials
        self.api_key: Optional[str] = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret
