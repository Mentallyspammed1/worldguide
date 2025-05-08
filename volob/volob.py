#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Pyrmethus's Unified Trading Familiar v1.1 (VOB Strategy)
# Combines Volumetric Order Block strategy with position management (TSL/TP/SL).

import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN, InvalidOperation, DivisionByZero
from typing import Dict, Optional, Any, Tuple, Union, List

# Third-party enchantments
import ccxt
from dotenv import load_dotenv
import pandas as pd # Keep pandas for strategy calculations
from colorama import init, Fore, Style, Back

# =============================================================================
# Initialization Rituals & Constants
# =============================================================================
init(autoreset=True)
getcontext().prec = 50 # Set precision for Decimal calculations

APP_NAME = "UnifiedTrader-VOB" # Indicate strategy in name
LOG_FORMAT_CONSOLE = f"{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.MAGENTA}{APP_NAME}{Style.RESET_ALL} - %(levelname)s - %(message)s"
LOG_FORMAT_FILE = "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- State Definitions ---
STATE_SEARCHING = "SEARCHING"
STATE_ENTERING = "ENTERING"
STATE_MANAGING_LONG = "MANAGING_LONG"
STATE_MANAGING_SHORT = "MANAGING_SHORT"
STATE_EXIT_TRIGGERED = "EXIT_TRIGGERED"
STATE_CLOSING = "CLOSING"
STATE_CLOSED = "CLOSED"
STATE_ERROR = "ERROR"
STATE_HALTED = "HALTED"

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path); print(Fore.CYAN + Style.DIM + f"# Secrets whispered from {dotenv_path}")
else: load_dotenv(); print(Fore.YELLOW + Style.BRIGHT + f"# Warning: .env file not found. Seeking secrets from environment variables.")

# =============================================================================
# Arcane Logging Configuration (Combined)
# =============================================================================
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
try: log_level = getattr(logging, log_level_str)
except AttributeError: print(f"{Fore.RED}Invalid LOG_LEVEL '{log_level_str}'. Defaulting to INFO."); log_level = logging.INFO

logger = logging.getLogger(APP_NAME)
logger.setLevel(log_level)
logger.propagate = False # Prevent double logging if root logger is configured

class ColorFormatter(logging.Formatter):
    """Custom formatter to add color to log output."""
    LOG_COLORS = { logging.DEBUG: Fore.CYAN + Style.DIM, logging.INFO: Fore.GREEN, logging.WARNING: Fore.YELLOW, logging.ERROR: Fore.RED, logging.CRITICAL: Fore.RED + Style.BRIGHT + Back.WHITE }
    RESET = Style.RESET_ALL
    def __init__(self, fmt=LOG_FORMAT_CONSOLE, datefmt=DATE_FORMAT): super().__init__(fmt, datefmt)
    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, Fore.WHITE)
        # Temporarily modify record for coloring
        original_levelname = record.levelname
        record.levelname = f"{log_color}{record.levelname:<8}{self.RESET}"
        # Add color to the message for warnings and errors
        if record.levelno >= logging.WARNING: record.msg = f"{log_color}{record.msg}{self.RESET}"
        formatted_message = super().format(record)
        # Restore original levelname
        record.levelname = original_levelname
        return formatted_message

# Add handlers only if they don't already exist
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(log_level); console_handler.setFormatter(ColorFormatter()); logger.addHandler(console_handler)
    log_file_path = os.getenv("LOG_FILE_PATH", f"{APP_NAME.lower()}.log")
    try:
        file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'); file_handler.setLevel(log_level); file_handler.setFormatter(logging.Formatter(LOG_FORMAT_FILE, datefmt=DATE_FORMAT)); logger.addHandler(file_handler)
        logger.info(f"{Fore.CYAN+Style.DIM}Logging detailed whispers to: {log_file_path}")
    except Exception as e: logger.error(f"{Fore.RED}Could not set up file logger at '{log_file_path}': {e}")

# =============================================================================
# Unified Trader Configuration Class
# =============================================================================
class TraderConfig:
    """Holds and validates parameters for the Unified Trader."""
    def __init__(self):
        logger.debug("Summoning unified configuration runes...")
        # --- Exchange & Symbol ---
        self.api_key: Optional[str] = os.getenv('BYBIT_API_KEY')
        self.api_secret: Optional[str] = os.getenv('BYBIT_API_SECRET')
        self.symbol: str = self._get_env('SYMBOL', 'BTC/USDT:USDT', cast_type=str)
        self.market_type: str = self._get_env('MARKET_TYPE', 'linear', cast_type=str, allowed_values=['linear', 'inverse']).lower()
        # --- Strategy & Timing ---
        self.interval: str = self._get_env('INTERVAL', '5m', cast_type=str) # VOB might need slightly higher timeframe
        self.ohlcv_limit: int = self._get_env('OHLCV_LIMIT', 200, cast_type=int, min_val=50, max_val=1000) # Need enough candles for pivots/EMA
        self.loop_sleep_seconds: int = self._get_env('LOOP_SLEEP_SECONDS', 30, cast_type=int, min_val=5) # Base loop interval when searching
        self.manage_interval_seconds: int = self._get_env('MANAGE_INTERVAL_SECONDS', 10, cast_type=int, min_val=3) # Faster checks when managing position
        # --- VOB Strategy Specific Parameters ---
        # VOB_VOLUME_EMA_SPAN: Period for the Exponential Moving Average of volume. Used to normalize volume and identify potentially high-volume candles.
        self.vob_volume_ema_span: int = self._get_env('VOB_VOLUME_EMA_SPAN', 20, cast_type=int, min_val=5)
        # VOB_PIVOT_WINDOW: The lookback/lookforward window size for identifying pivot highs/lows. A candle is a pivot if it's the highest/lowest in this window (centered). Odd number is recommended.
        self.vob_pivot_window: int = self._get_env('VOB_PIVOT_WINDOW', 11, cast_type=int, min_val=3)
        # VOB_SL_BUFFER_PCT: Percentage buffer added to the stop-loss price relative to the triggering order block's edge. Helps avoid premature stops.
        self.vob_sl_buffer_pct: Decimal = self._get_env('VOB_SL_BUFFER_PCT', '0.001', cast_type=Decimal, min_val=Decimal("0.0001"), max_val=Decimal("0.05")) # % buffer for SL (0.1%)
        # --- Risk & Sizing ---
        self.leverage: int = self._get_env('LEVERAGE', 10, cast_type=int, min_val=1, max_val=125)
        self.risk_percentage: Decimal = self._get_env('RISK_PERCENTAGE', '0.01', cast_type=Decimal, min_val=Decimal("0.0001"), max_val=Decimal("0.1")) # Risk per trade relative to wallet balance
        # --- Position Management (Exit) ---
        self.trailing_stop_pct: Decimal = self._get_env('TRAILING_STOP_PCT', '0.02', cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("0.99")) # 2% TSL
        self.take_profit_pct: Optional[Decimal] = self._get_env('TAKE_PROFIT_PCT', '0.04', cast_type=Decimal, required=False, min_val=Decimal("0.001"), max_val=Decimal("10.0")) # 4% TP (Optional)
        # --- Execution ---
        self.entry_order_type: str = self._get_env('ENTRY_ORDER_TYPE', 'market', cast_type=str, allowed_values=['market', 'limit'])
        self.exit_order_type: str = self._get_env('EXIT_ORDER_TYPE', 'market', cast_type=str, allowed_values=['market']) # Market exit is generally safest/fastest
        self.order_check_delay_seconds: int = self._get_env('ORDER_CHECK_DELAY_SECONDS', 2, cast_type=int, min_val=1)
        self.order_check_attempts: int = self._get_env('ORDER_CHECK_ATTEMPTS', 5, cast_type=int, min_val=1, max_val=20)
        self.max_fetch_retries: int = self._get_env('MAX_FETCH_RETRIES', 3, cast_type=int, min_val=1, max_val=10)
        self.dry_run: bool = self._get_env('DRY_RUN', True, cast_type=bool) # Safety first!
        self._validate()
        logger.debug("Unified configuration runes validated.")

    def _get_env(self, key: str, default: Any = None, cast_type: type = str, required: bool = False,
                 min_val: Optional[Union[int, float, Decimal]] = None,
                 max_val: Optional[Union[int, float, Decimal]] = None,
                 allowed_values: Optional[List[str]] = None) -> Any:
        """Fetches and casts environment variable, with validation and logging."""
        value_str = os.getenv(key)
        is_secret = "SECRET" in key or "KEY" in key # Simple check for secrets
        log_value = "****" if is_secret and value_str else value_str

        if value_str is None or value_str.strip() == "":
            if required:
                logger.critical(f"{Fore.RED+Style.BRIGHT}Required config '{key}' is missing.")
                raise ValueError(f"Required config '{key}' is missing.")
            if default is None:
                logger.debug(f"Config '{key}' not found and not required, returning None.")
                return None
            value_str = str(default)
            log_default = "****" if is_secret else default
            logger.debug(f"Config '{key}' not found, using default: {log_default}")
        else:
             logger.debug(f"Found config '{key}': {log_value}")

        try:
            if cast_type == bool:
                value = value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                 # Handle potential errors during Decimal conversion
                 value = Decimal(value_str)
            elif cast_type == int:
                # Convert via float first to handle scientific notation if needed
                value = int(float(value_str))
            elif cast_type == float:
                value = float(value_str)
            else: # Default to string
                value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Cast error for '{key}' ('{value_str}') to {cast_type.__name__}: {e}")
            raise ValueError(f"Cast error for '{key}' ('{value_str}') to {cast_type.__name__}: {e}") from e

        if allowed_values and value not in allowed_values:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Invalid value '{value}' for '{key}'. Allowed: {allowed_values}.")
            raise ValueError(f"Invalid value '{value}' for '{key}'. Allowed: {allowed_values}.")

        # Numeric range validation
        if isinstance(value, (int, float, Decimal)):
            try:
                # Ensure comparison types match the cast_type
                min_val_comp = cast_type(min_val) if min_val is not None else None
                max_val_comp = cast_type(max_val) if max_val is not None else None

                if min_val_comp is not None and value < min_val_comp:
                    logger.critical(f"{Fore.RED+Style.BRIGHT}Config '{key}' value {value} is below minimum allowed {min_val}.")
                    raise ValueError(f"Config '{key}' value {value} is below minimum allowed {min_val}.")
                if max_val_comp is not None and value > max_val_comp:
                    logger.critical(f"{Fore.RED+Style.BRIGHT}Config '{key}' value {value} is above maximum allowed {max_val}.")
                    raise ValueError(f"Config '{key}' value {value} is above maximum allowed {max_val}.")
            except (ValueError, TypeError, InvalidOperation) as e:
                 logger.critical(f"{Fore.RED+Style.BRIGHT}Validation range comparison error for '{key}': {e}")
                 raise ValueError(f"Validation range comparison error for '{key}': {e}") from e

        return value

    def _validate(self):
        """Performs final validation checks on combined configuration."""
        if not self.dry_run and (not self.api_key or not self.api_secret):
            raise ValueError("API Key/Secret required when DRY_RUN is False.")
        if '/' not in self.symbol or ':' not in self.symbol:
            raise ValueError(f"Invalid SYMBOL format: '{self.symbol}'. Expected format like 'BASE/QUOTE:SETTLEMENT'.")
        if self.vob_pivot_window % 2 == 0:
            logger.warning(f"{Fore.YELLOW}VOB_PIVOT_WINDOW ({self.vob_pivot_window}) is even. Pivot calculation uses a centered window, which is typically more intuitive with an odd window size. Logic should still work but may be slightly off-center.")
        # Add more checks here if needed (e.g., interval format)
        logger.info(f"{Fore.GREEN}Unified configuration runes successfully summoned and validated.")

# =============================================================================
# The Unified Trading Familiar Class (with VOB Strategy)
# =============================================================================
class UnifiedTrader:
    """Orchestrates trading strategy execution and position management using VOB."""

    def __init__(self, config: TraderConfig):
        self.config = config
        self.state = STATE_SEARCHING
        self.exchange = self._initialize_exchange()
        self.market_info = self._load_market_info()
        # Set Decimal quantization based on market precision
        self.price_quantizer = Decimal(f'1e-{self.market_info["precision"]["price"]}')
        self.amount_quantizer = Decimal(f'1e-{self.market_info["precision"]["amount"]}')
        # Ensure min_order_amount is Decimal
        self.min_order_amount = Decimal(str(self.market_info['limits']['amount']['min']))
        self.base_currency = self.market_info['base']
        self.quote_currency = self.market_info['quote']
        self._set_leverage()
        self.active_position: Dict[str, Any] = {} # Stores details of the current open position

    # --- Core Exchange Interaction & Utility Methods ---
    # These methods handle communication with the exchange, data fetching,
    # quantization, sizing, and order management. They are designed to be
    # strategy-agnostic and are reused from the base UnifiedTrader.

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes the CCXT exchange instance."""
        logger.info(f"{Fore.CYAN}Initializing connection to Bybit ({self.config.market_type})...")
        try:
            exchange = ccxt.bybit({
                'apiKey': self.config.api_key if not self.config.dry_run else None,
                'secret': self.config.api_secret if not self.config.dry_run else None,
                'enableRateLimit': True, # Respect exchange rate limits
                'options': {
                    'defaultType': 'swap', # Often required for futures/perpetuals
                    'defaultSubType': self.config.market_type, # 'linear' or 'inverse'
                    'adjustForTimeDifference': True # Sync time with exchange
                },
                'hostname': 'api.bybit.com' # Explicitly set hostname if needed for specific exchanges/regions
            })
            if not self.config.dry_run:
                 # Fetch balance or account info to test credentials and connection
                 exchange.fetch_balance()
                 logger.info(f"{Fore.GREEN}Exchange connection established and credentials verified.")
            else:
                 logger.warning(f"{Fore.YELLOW}Dry run mode: Skipping credential verification.")
                 logger.info(f"{Fore.GREEN}Exchange instance created for dry run.")
            return exchange
        except ccxt.AuthenticationError as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Authentication Error: Check your API Key and Secret. {e}")
            raise SystemExit("Auth Failed") from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Exchange Connection Error: Could not connect or communicate with the exchange. {e}")
            raise SystemExit("Exchange Connect Failed") from e
        except Exception as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Unexpected Error during Exchange Initialization: {e}", exc_info=True)
            raise SystemExit("Exchange Init Failed") from e

    def _load_market_info(self) -> Dict:
        """Loads market information for the trading symbol."""
        logger.info(f"{Fore.CYAN}Loading market data for {self.config.symbol}...")
        try:
            # Fetch markets information, force reload to get the latest data
            markets = self.exchange.load_markets(True)
            if self.config.symbol not in markets:
                logger.critical(f"{Fore.RED+Style.BRIGHT}Symbol Error: Trading symbol '{self.config.symbol}' not found on the exchange.")
                raise ccxt.BadSymbol(f"Symbol '{self.config.symbol}' not found.")
            market = markets[self.config.symbol]
            # Log key market details for verification
            logger.info(f"{Fore.GREEN}Market data loaded successfully.")
            logger.info(f"  Symbol: {market['symbol']}")
            logger.info(f"  Base: {market['base']}, Quote: {market['quote']}")
            logger.info(f"  Price Precision: {market['precision']['price']}")
            logger.info(f"  Amount Precision: {market['precision']['amount']}")
            logger.info(f"  Min Amount: {market['limits']['amount']['min']}")
            return market
        except ccxt.BadSymbol as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}{e}")
            raise SystemExit("Invalid Symbol") from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Market Data Load Error: Could not fetch market information. {e}")
            raise SystemExit("Market Load Failed") from e
        except Exception as e:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Unexpected Error during Market Data Loading: {e}", exc_info=True)
            raise SystemExit("Market Load Failed") from e

    def _set_leverage(self):
        """Sets the desired leverage for the trading symbol."""
        if self.config.leverage:
            logger.info(f"{Fore.CYAN}Attempting to set leverage for {self.config.symbol} to {self.config.leverage}x...")
            if self.config.dry_run and (not self.config.api_key or not self.config.api_secret):
                 logger.warning(f"{Fore.YELLOW}Dry run: Cannot set leverage without API keys. Skipping actual leverage setting.")
                 return
            try:
                # Check if the exchange supports setting leverage via API
                if self.exchange.has['setLeverage']:
                    # Some exchanges require market type or other params for leverage
                    # Bybit set_leverage usually requires symbol and leverage
                    response = self.exchange.set_leverage(self.config.leverage, self.config.symbol)
                    logger.info(f"{Fore.GREEN}Leverage set request successful.")
                    logger.debug(f"Set leverage response: {response}")
                else:
                    logger.warning(f"{Fore.YELLOW}Exchange does not support setting leverage via API for {self.config.symbol}. Please set leverage manually on the exchange platform.")
            except ccxt.AuthenticationError:
                logger.error(f"{Fore.RED}Authentication failed while trying to set leverage.")
            except ccxt.ExchangeError as e:
                # This can fail if there are open positions or orders, or if leverage is outside allowed range
                logger.error(f"{Fore.RED}Failed to set leverage on the exchange: {e}")
            except Exception as e:
                logger.error(f"{Fore.RED}Unexpected error setting leverage: {e}", exc_info=True)
        else:
            logger.info(f"{Fore.CYAN}Leverage setting skipped as LEVERAGE config is 0 or None.")


    def _quantize_price(self, price: Decimal) -> Decimal:
        """Quantizes a price to the symbol's price precision."""
        # Use ROUND_HALF_UP for prices
        return price.quantize(self.price_quantizer, rounding=ROUND_HALF_UP)

    def _quantize_amount(self, amount: Decimal) -> Decimal:
        """Quantizes an amount (size) to the symbol's amount precision."""
        # Use ROUND_DOWN for amounts to avoid exceeding available balance/margin
        return amount.quantize(self.amount_quantizer, rounding=ROUND_DOWN)

    def _fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data for the configured symbol and interval."""
        logger.debug(f"Fetching OHLCV for {self.config.symbol} ({self.config.interval}, limit={self.config.ohlcv_limit})...")
        for attempt in range(self.config.max_fetch_retries):
            try:
                # fetch_ohlcv returns a list of lists: [[timestamp, open, high, low, close, volume], ...]
                ohlcv = self.exchange.fetch_ohlcv(self.config.symbol, timeframe=self.config.interval, limit=self.config.ohlcv_limit)
                if not ohlcv:
                    logger.warning(f"Empty OHLCV data received from exchange (Attempt {attempt+1})."); time.sleep(1*(attempt+1)); continue

                # Convert to pandas DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Convert numerical columns to Decimal for precision
                num_cols = ['open', 'high', 'low', 'close', 'volume']
                # Use applymap for element-wise application
                df[num_cols] = df[num_cols].applymap(lambda x: Decimal(str(x)) if pd.notna(x) else None)

                # Drop rows with any NaN in the numerical columns after conversion
                df.dropna(subset=num_cols, inplace=True)

                if df.empty:
                    logger.warning(f"OHLCV data empty after processing (Attempt {attempt+1})."); time.sleep(1*(attempt+1)); continue

                logger.debug(f"Successfully fetched and processed {len(df)} OHLCV records.")
                return df

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit exceeded while fetching OHLCV (Attempt {attempt+1}). Waiting before retry... ({e})"); time.sleep(self.exchange.rateLimit/1000*(attempt+2))
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Network/Exchange error fetching OHLCV (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(2**attempt) # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error fetching OHLCV (Attempt {attempt+1}): {e}", exc_info=True); time.sleep(3**attempt) # Longer backoff for unexpected errors

        logger.error(f"{Fore.RED+Style.BRIGHT}Failed to fetch OHLCV after {self.config.max_fetch_retries} attempts.")
        return None

    def _fetch_current_price(self) -> Optional[Decimal]:
        """Fetches the current market price (last trade price or close from ticker)."""
        logger.debug(f"Fetching ticker for {self.config.symbol} to get current price...")
        for attempt in range(self.config.max_fetch_retries):
            try:
                ticker = self.exchange.fetch_ticker(self.config.symbol)
                # Use 'last' or 'close' price from ticker data
                price_str = ticker.get('last') or ticker.get('close')
                if price_str is None:
                     logger.warning(f"Ticker data missing 'last' or 'close' price (Attempt {attempt+1}). Data: {ticker}."); time.sleep(1*(attempt+1)); continue
                price = Decimal(str(price_str))
                logger.debug(f"Current price fetched: {price:.{self.market_info['precision']['price']}f}")
                return price
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit fetching ticker (Attempt {attempt+1}). Wait... ({e})"); time.sleep(self.exchange.rateLimit/1000*(attempt+2))
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                logger.warning(f"Network/Exchange error fetching ticker (Attempt {attempt+1}): {e}. Retry..."); time.sleep(2**attempt)
            except Exception as e:
                logger.error(f"Unexpected error fetching ticker (Attempt {attempt+1}): {e}", exc_info=True); time.sleep(3**attempt)
        logger.error(f"{Fore.RED+Style.BRIGHT}Failed to fetch price after {self.config.max_fetch_retries} attempts.")
        return None

    def _get_balance(self, currency: str) -> Optional[Decimal]:
        """Fetches the available balance for a given currency."""
        if self.config.dry_run:
            # Simulate a reasonable balance in dry run
            logger.debug(f"Dry run: Simulating available balance for {currency}.")
            # Use a fixed large amount or read from a dummy config if needed
            return Decimal("10000.0")

        logger.debug(f"Fetching balance for {currency}...")
        try:
            balance_data = self.exchange.fetch_balance()
            free_balance = Decimal("0.0")

            # Exchange-specific parsing might be needed. CCXT standardizes much,
            # but details in the raw 'info' can vary. This is a common pattern
            # for unified accounts or specific sub-accounts.
            # Example for Bybit Unified Trading Account:
            if 'info' in balance_data and 'result' in balance_data['info'] and 'list' in balance_data['info']['result']:
                 for account in balance_data['info']['result']['list']:
                     # Check account type if necessary (e.g., 'UNIFIED', 'CONTRACT', 'SPOT')
                     # For perpetuals, usually 'CONTRACT' or 'UNIFIED' is relevant
                     # if account.get('accountType') in ['UNIFIED', 'CONTRACT']: # Example filter
                         coin_list = account.get('coin', [])
                         for coin_data in coin_list:
                             if coin_data.get('coin') == currency:
                                 # Use 'availableToWithdraw' or 'walletBalance' depending on context/exchange API
                                 # 'availableToWithdraw' is often the amount usable for new trades/withdrawals
                                 balance_str = coin_data.get('availableToWithdraw', coin_data.get('walletBalance', '0'))
                                 try:
                                     free_balance = Decimal(str(balance_str))
                                     # If found and positive, we can often stop searching
                                     if free_balance > 0:
                                          logger.info(f"Available {currency} balance found: {free_balance}")
                                          return free_balance
                                 except InvalidOperation:
                                     logger.warning(f"Could not parse balance for {currency}: '{balance_str}'")
                                 break # Found the currency in this account entry
            elif currency in balance_data and balance_data[currency] and 'free' in balance_data[currency]:
                # Fallback to standard CCXT structure
                free_balance = Decimal(str(balance_data[currency]['free']))
                logger.info(f"Available {currency} balance (standard): {free_balance}")
                return free_balance
            else:
                logger.warning(f"Could not find balance for {currency} in standard or known 'info' structure.")


            logger.warning(f"{Fore.YELLOW}Available {currency} balance is zero or could not be determined.")
            return Decimal("0.0") # Return 0 if not found or parsed

        except ccxt.AuthenticationError:
            logger.error(f"{Fore.RED}Authentication failed while fetching balance.")
            return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(f"{Fore.RED}Error fetching balance from exchange: {e}")
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error fetching balance: {e}", exc_info=True)
            return None


    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """Calculates the desired position size based on risk percentage and SL distance."""
        logger.debug("Calculating position size...")

        wallet_balance = self._get_balance(self.quote_currency)
        if wallet_balance is None or wallet_balance <= Decimal("0"):
            logger.error(f"{Fore.RED}Cannot calculate size: Available wallet balance is zero or could not be fetched.")
            return None

        if entry_price <= Decimal("0") or stop_loss_price <= Decimal("0"):
            logger.error(f"{Fore.RED}Cannot calculate size: Invalid entry ({entry_price}) or stop loss ({stop_loss_price}) price.")
            return None

        try:
            # Calculate the value of 1 unit of the base currency in quote currency at entry price
            # For linear (e.g., BTC/USDT), 1 BTC = entry_price USDT
            # For inverse (e.g., BTC/USD), 1 Contract might be 1 USD, or 1 BTC, depends on symbol.
            # CCXT 'linear' means quote-settled, 'inverse' means base-settled.
            # For linear, amount is in base currency (e.g., BTC amount).
            # For inverse, amount is often in contracts, representing a fixed quantity of the quote asset (e.g., USD) or base asset.
            # Bybit inverse BTCUSD is settled in BTC, 1 contract is 1 USD.
            # Let's assume amount is in base currency for linear, and contracts (representing fixed quote value, e.g., USD) for inverse.
            # This needs careful handling per exchange/symbol if the 'amount' unit varies.
            # For Bybit linear (BTC/USDT:USDT), amount is BTC. Notional = amount * entry_price.
            # For Bybit inverse (BTC/USD:BTC), amount is contracts (value 1 USD). Notional = amount. Position in BTC = amount / entry_price.

            price_diff = abs(entry_price - stop_loss_price)
            if price_diff < entry_price * Decimal("0.00001"): # Check if price diff is effectively zero (e.g., less than 0.001% of price)
                 logger.warning(f"{Fore.YELLOW}Cannot calculate size: Entry price ({entry_price}) and SL ({stop_loss_price}) are too close. Risk is infinite.")
                 return None

            # Risk amount in quote currency (e.g., USDT)
            risk_amount_quote = wallet_balance * self.config.risk_percentage

            if self.config.market_type == 'linear':
                # Linear market: Position size is in base currency (e.g., BTC)
                # Risk_amount_quote = Position_size_base * price_diff
                # Position_size_base = Risk_amount_quote / price_diff
                position_size_base = risk_amount_quote / price_diff
                order_size = position_size_base
                # Notional value is the total value of the position at entry price
                notional_value = order_size * entry_price
                amount_unit = self.base_currency # e.g., BTC

            elif self.config.market_type == 'inverse':
                 # Inverse market (e.g., BTC/USD settled in BTC): Position size is in contracts.
                 # The value of 1 contract is typically fixed (e.g., 1 USD).
                 # PnL for inverse is often calculated as: PnL_BTC = contracts * (1/Entry_Price - 1/Exit_Price)
                 # Risk in BTC = contracts * (1/SL_Price - 1/Entry_Price) (for long)
                 # Risk in Quote (USD) = Risk_in_BTC * Current_Price (approx)
                 # This calculation is more complex and depends heavily on the specific inverse contract details.
                 # A simplified approach assumes risk is calculated based on notional value in quote currency (USD).
                 # If 1 contract = 1 USD (common inverse setup), then Notional = amount_in_contracts.
                 # Position value in BTC = Notional / Price = amount_in_contracts / entry_price
                 # Risk_amount_quote = |Entry_Price - SL_Price| * Position_size_base
                 # Risk_amount_quote = |Entry_Price - SL_Price| * (amount_in_contracts / entry_price)
                 # amount_in_contracts = (Risk_amount_quote * entry_price) / |Entry_Price - SL_Price|

                 # Let's use the formula assuming amount is in contracts and Notional = amount_in_contracts (for USD-margined inverse)
                 # If it's BTC-margined inverse (BTC/USD:BTC), amount is still contracts, but Notional calculation might differ slightly or be based on BTC value.
                 # The formula (Risk_amount_quote * entry_price) / price_diff is standard for calculating base currency size in linear.
                 # For inverse (USD-margined), amount_in_contracts seems equivalent to notional in USD if 1 contract = 1 USD.
                 # The formula (wallet_balance * risk_percentage * entry_price) / price_diff calculates the notional value (in quote currency) you can trade.
                 # If 1 contract is 1 USD, this notional value *is* the number of contracts.
                 position_notional_quote = (wallet_balance * self.config.risk_percentage * entry_price) / price_diff
                 order_size = position_notional_quote # Assuming 1 contract = 1 quote currency unit value
                 notional_value = order_size # Notional value is the number of contracts if 1 contract = 1 USD
                 amount_unit = 'Contracts' # e.g., Contracts

                 # Note: This inverse sizing formula assumes 1 contract has a value of 1 unit of the quote currency (e.g., 1 USD).
                 # If the contract value is different (e.g., BTC-margined contracts representing 100 USD), the calculation needs adjustment.
                 # For Bybit BTC/USD (inverse perpetual), 1 contract = 1 USD. So this formula should work.

            else:
                 logger.error(f"{Fore.RED}Unsupported market type '{self.config.market_type}' for sizing.")
                 return None

            # Quantize the calculated size
            final_order_size = self._quantize_amount(order_size)

            # Check against minimum order amount
            if final_order_size < self.min_order_amount:
                logger.warning(f"{Fore.YELLOW}Calculated size ({final_order_size}) is below the minimum allowed order amount ({self.min_order_amount}). Cannot place trade.")
                return None

            # Estimate required initial margin (simplified)
            # Initial Margin = Notional Value / Leverage
            # This is a simplification; actual margin calculation depends on exchange rules, risk limits, etc.
            required_margin = notional_value / Decimal(self.config.leverage)
            if required_margin > wallet_balance:
                 logger.warning(f"{Fore.YELLOW}Insufficient margin. Estimated required margin for this trade ({required_margin:.4f} {self.quote_currency}) exceeds available balance ({wallet_balance:.4f} {self.quote_currency}). Cannot place trade.")
                 return None

            logger.info(f"Calculated position size: {final_order_size} {amount_unit}")
            logger.info(f"  Risk Amount: {risk_amount_quote:.4f} {self.quote_currency}")
            logger.info(f"  Estimated Notional: {notional_value:.4f} {self.quote_currency}")
            logger.info(f"  Estimated Margin Req: {required_margin:.4f} {self.quote_currency}")

            return final_order_size

        except (InvalidOperation, DivisionByZero) as e:
            logger.error(f"{Fore.RED}Math error during position sizing calculation: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error during position sizing: {e}", exc_info=True)
            return None


    def _place_order(self, side: str, amount: Decimal, order_type: str, price: Optional[Decimal]=None, params: Optional[Dict]=None) -> Optional[Dict]:
        """Places an order on the exchange."""
        # Validate inputs
        if side not in ['buy', 'sell']: logger.error(f"{Fore.RED}Invalid order side: {side}"); return None
        if amount is None or amount <= Decimal('0'): logger.error(f"{Fore.RED}Invalid order amount: {amount}"); return None
        if order_type not in ['market', 'limit']: logger.error(f"{Fore.RED}Invalid order type: {order_type}"); return None
        if order_type == 'limit' and (price is None or price <= Decimal('0')): logger.error(f"{Fore.RED}Limit order requires a valid price."); return None

        # Quantize amount and price
        quantized_amount = self._quantize_amount(amount)
        quantized_price = self._quantize_price(price) if price is not None else None

        if quantized_amount < self.min_order_amount:
             logger.error(f"{Fore.RED}Order amount {quantized_amount} is below minimum order size {self.min_order_amount}.")
             return None
        if quantized_price is not None and quantized_price <= Decimal('0'):
             logger.error(f"{Fore.RED}Quantized price resulted in zero or negative value: {quantized_price}")
             return None


        if self.config.dry_run:
            order_id = f"DRYRUN_{side.upper()}_{int(time.time())}"
            logger.warning(f"{Back.YELLOW+Fore.BLACK} DRY RUN {Style.RESET_ALL}{Fore.YELLOW} Simulating {order_type} order: {side.upper()} {quantized_amount} {self.config.symbol} @ {quantized_price if quantized_price is not None else 'Market'}")
            # Simulate order structure return
            simulated_order = {
                'id': order_id,
                'status': 'closed' if order_type == 'market' else 'open', # Market orders are instantly 'closed' in simulation
                'symbol': self.config.symbol,
                'type': order_type,
                'side': side,
                'amount': float(quantized_amount),
                'price': float(quantized_price) if quantized_price is not None else None,
                'filled': float(quantized_amount) if order_type == 'market' else 0.0,
                'remaining': 0.0 if order_type == 'market' else float(quantized_amount),
                'cost': float(quantized_amount) * float(quantized_price if quantized_price is not None else (self._fetch_current_price() or Decimal('0'))) if order_type == 'market' else 0.0, # Rough cost estimate
                'average': float(quantized_price) if quantized_price is not None else float(self._fetch_current_price() or Decimal('0')), # Rough average fill price
                'datetime': self.exchange.iso8601(self.exchange.milliseconds()),
                'timestamp': self.exchange.milliseconds(),
                'info': {'dry_run': True, 'simulated': True, 'orderId': order_id} # Add custom info
            }
            logger.info(f"{Back.YELLOW+Fore.BLACK} DRY RUN {Style.RESET_ALL}{Fore.YELLOW} Simulated order ID: {order_id}")
            return simulated_order


        logger.info(f"{Fore.BLUE+Style.BRIGHT}Attempting to place {order_type} order: {side.upper()} {quantized_amount} {self.config.symbol}...")
        order_params = params if params else {}
        # Pass float values to ccxt
        ccxt_amount = float(quantized_amount)
        ccxt_price = float(quantized_price) if quantized_price is not None else None

        for attempt in range(self.config.max_fetch_retries): # Re-using fetch retries for order placement
            try:
                order = self.exchange.create_order(self.config.symbol, order_type, side, ccxt_amount, ccxt_price, params=order_params)
                logger.info(f"{Fore.GREEN+Style.BRIGHT}Order placed successfully! ID: {order.get('id')}")
                logger.debug(f"Order details: {order}")
                return order
            except ccxt.InsufficientFunds as e:
                logger.error(f"{Fore.RED+Style.BRIGHT}Order Failed - Insufficient Funds: {e}")
                return None # Critical failure, stop retrying
            except ccxt.InvalidOrder as e:
                 logger.error(f"{Fore.RED+Style.BRIGHT}Order Failed - Invalid Order: {e}. Check parameters, price, amount, min/max limits, etc.")
                 return None # Critical failure, stop retrying
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit placing order (Attempt {attempt+1}). Waiting... ({e})"); time.sleep(self.exchange.rateLimit/1000*(attempt+2))
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Network/Exchange error placing order (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(self.config.order_check_delay_seconds*(attempt+1)) # Use check delay for order retries
            except Exception as e:
                logger.error(f"{Fore.RED}Unexpected error placing order (Attempt {attempt+1}): {e}", exc_info=True); time.sleep(self.config.order_check_delay_seconds*(attempt+2))

        logger.error(f"{Fore.RED+Style.BRIGHT}Failed to place order after {self.config.max_fetch_retries} attempts.")
        return None

    def _confirm_order_status(self, order_id: str, target_status: str = 'closed') -> bool:
        """Waits and confirms if an order reaches a target status."""
        if self.config.dry_run:
            logger.info(f"{Fore.YELLOW}Dry run: Assuming order {order_id} reached status '{target_status}'.")
            return True # In dry run, we assume success for confirmation checks

        if not order_id:
            logger.error(f"{Fore.RED}Cannot confirm order status: No Order ID provided.")
            return False

        logger.info(f"Confirming status '{target_status}' for Order ID: {order_id}...")

        for attempt in range(self.config.order_check_attempts):
            try:
                logger.debug(f"Fetching order status for {order_id} (Attempt {attempt+1})...")
                order_data = self.exchange.fetch_order(order_id, self.config.symbol)
                status = order_data.get('status')

                logger.info(f"  Order Status: {status}")

                if status == target_status:
                    logger.info(f"{Fore.GREEN}Order {order_id} successfully confirmed as '{status}'.")
                    return True
                if status in ['canceled', 'rejected', 'expired', 'closed']: # If closed but target wasn't closed, means it filled. If target was closed, success.
                     if target_status != 'closed' and status == 'closed':
                         logger.info(f"{Fore.GREEN}Order {order_id} confirmed '{status}' (filled).")
                         return True # Order filled even if target was 'open' (e.g. limit order)
                     elif target_status == 'closed' and status == 'closed':
                          logger.info(f"{Fore.GREEN}Order {order_id} confirmed '{status}'.")
                          return True
                     else:
                         logger.error(f"{Fore.RED}Order {order_id} failed to reach target status '{target_status}', ended as '{status}'.")
                         return False # Order ended in a final state other than the target

                # If not in target or failure state, wait and retry
                logger.info(f"Status is '{status}'. Waiting {self.config.order_check_delay_seconds}s before retry (Attempt {attempt+1}/{self.config.order_check_attempts})...")
                time.sleep(self.config.order_check_delay_seconds)

            except ccxt.OrderNotFound:
                logger.error(f"{Fore.RED}Order {order_id} not found on exchange (Attempt {attempt+1}). It might have filled very quickly or failed immediately.")
                # Sometimes market orders disappear quickly after filling. Wait a moment and try fetching again or assume filled if dry_run==False?
                # Given the retry logic, if it's truly not found after a few tries, something is likely wrong.
                time.sleep(self.config.order_check_delay_seconds) # Wait before declaring failure
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Network/Exchange error confirming order status (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(self.config.order_check_delay_seconds)
            except Exception as e:
                logger.error(f"{Fore.RED}Unexpected error confirming order status (Attempt {attempt+1}): {e}", exc_info=True); time.sleep(self.config.order_check_delay_seconds)

        logger.error(f"{Fore.RED+Style.BRIGHT}Failed to confirm status '{target_status}' for order {order_id} after {self.config.order_check_attempts} attempts.")
        return False

    def _get_actual_entry_price(self, order_id: str) -> Optional[Decimal]:
         """Fetches the average fill price for a closed order."""
         if self.config.dry_run:
             # In dry run, we don't have an actual fill price from the exchange.
             # The estimated price used for sizing is the best we have.
             logger.info(f"{Fore.YELLOW}Dry run: Cannot fetch actual fill price. Using estimated entry price.")
             # Return None, _handle_searching will fall back to estimate
             return None

         if not order_id:
             logger.error(f"{Fore.RED}Cannot fetch actual entry price: No Order ID provided.")
             return None

         logger.info(f"Fetching actual fill price for order {order_id}...")
         try:
             # Retry fetching the order details as it might take a moment for fill details to update
             for attempt in range(self.config.order_check_attempts): # Use same attempt count as status check
                 order_data = self.exchange.fetch_order(order_id, self.config.symbol)
                 # Check for 'average' price which represents the average fill price for potentially partial fills
                 if order_data and order_data.get('average') is not None and order_data.get('average') > 0:
                     avg_price = Decimal(str(order_data['average']))
                     logger.info(f"{Fore.GREEN}Found actual average fill price for order {order_id}: {avg_price:.{self.market_info['precision']['price']}f}")
                     return avg_price
                 elif order_data and order_data.get('status') == 'closed' and order_data.get('filled', 0) > 0:
                     # Fallback: If order is closed and filled but 'average' is missing,
                     # sometimes 'price' is the market price for market orders.
                     # This is less reliable than 'average' but better than nothing if needed.
                     # Check if filled amount matches total amount to be sure it's fully filled.
                     if order_data.get('filled') >= order_data.get('amount', 0) * 0.999: # Allow for tiny precision differences
                         price_fallback = order_data.get('price')
                         if price_fallback is not None and price_fallback > 0:
                             logger.warning(f"{Fore.YELLOW}Using order 'price' as fallback fill price for {order_id} ({price_fallback}) as 'average' is missing/zero.")
                             return Decimal(str(price_fallback))
                     else:
                          logger.debug(f"Order {order_id} status closed, but not fully filled yet? Filled: {order_data.get('filled')}, Amount: {order_data.get('amount')}")


                 # If order is not yet closed or filled, wait and retry
                 if order_data and order_data.get('status') in ['open', 'partially_filled']:
                     logger.debug(f"Order {order_id} not yet fully filled (Status: {order_data.get('status')}). Waiting {self.config.order_check_delay_seconds}s...")
                     time.sleep(self.config.order_check_delay_seconds)
                 else:
                      # Order might be canceled/rejected or status unknown, or closed with zero fill
                      logger.warning(f"Could not determine average fill price for order {order_id}. Status: {order_data.get('status')}. Data: {order_data}. Attempt {attempt+1}/{self.config.order_check_attempts}")
                      if order_data and order_data.get('status') in ['canceled', 'rejected', 'expired']:
                          logger.error(f"{Fore.RED}Order {order_id} failed ({order_data.get('status')}). Cannot get fill price.")
                          return None # Order failed, no fill price

             logger.error(f"{Fore.RED}Failed to determine actual fill price for order {order_id} after {self.config.order_check_attempts} attempts.")
             return None

         except ccxt.OrderNotFound:
             logger.error(f"{Fore.RED}Order {order_id} not found when trying to get fill price. It might have been a dry run order or a very old/invalid ID.")
             return None
         except Exception as e:
              logger.error(f"{Fore.RED}Error fetching fill price for {order_id}: {e}", exc_info=True)
              return None


    # --- VOB Strategy Specific Methods ---

    def _calculate_volumetric_order_blocks(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Identifies potential support (bull) and resistance (bear) zones based on
        pivot points and volume characteristics. Integrated from VOB script.
        Returns the analyzed DataFrame and a dictionary of identified blocks.
        """
        if df is None or df.empty:
            logger.error(f"{Fore.RED}VOB Calc: Input DataFrame is empty.")
            return None, None

        logger.debug(f"{Fore.CYAN}Calculating volumetric order blocks...")
        df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame
        ema_span = self.config.vob_volume_ema_span
        pivot_window = self.config.vob_pivot_window

        # Ensure pivot window is valid and we have enough data
        if pivot_window < 3 or pivot_window > len(df_copy):
             logger.warning(f"{Fore.YELLOW}Invalid VOB_PIVOT_WINDOW size ({pivot_window}) or not enough data ({len(df_copy)}). Must be between 3 and len(df). Cannot calculate VOBs.")
             # Return the df with potential volume columns added if possible, but no blocks
             if len(df_copy) >= ema_span:
                  df_copy['volume_ema'] = df_copy['volume'].ewm(span=ema_span, adjust=False).mean()
                  df_copy['volume_norm'] = df_copy['volume'] / (df_copy['volume_ema'] + Decimal("1e-18"))
             return df_copy, None

        # Ensure pivot_window is odd for a perfectly centered window check
        if pivot_window % 2 == 0:
             # Adjust window slightly or warn. Warning is handled in config validation.
             # The rolling window with center=True handles even windows by default (slightly off-center)
             pass # Warning already issued by config validator

        half_window = pivot_window // 2 # Used for centered window logic indices

        try:
            # --- Volume analysis ---
            # Calculate EMA of volume
            df_copy['volume_ema'] = df_copy['volume'].ewm(span=ema_span, adjust=False).mean()
            # Calculate normalized volume (volume relative to its EMA) - indicates unusual volume
            # Add a small epsilon to the denominator to prevent division by zero if EMA is 0
            df_copy['volume_norm'] = df_copy['volume'] / (df_copy['volume_ema'] + Decimal("1e-18"))

            # --- Identify pivot points ---
            # A pivot high is a candle whose high is the highest within the rolling window (including itself)
            # A pivot low is a candle whose low is the lowest within the rolling window (including itself)
            # Use center=True to have the window centered around the current candle.
            # min_periods=pivot_window ensures we only calculate pivots once the window is full.
            # apply(lambda x: x.iloc[half_window] == x.max(), raw=True) efficiently checks if the center element is the max/min
            df_copy['pivot_high'] = df_copy['high'].rolling(window=pivot_window, center=True, min_periods=pivot_window).apply(lambda x: x.iloc[half_window] == x.max(), raw=True).fillna(0)
            df_copy['pivot_low'] = df_copy['low'].rolling(window=pivot_window, center=True, min_periods=pivot_window).apply(lambda x: x.iloc[half_window] == x.min(), raw=True).fillna(0)

            bull_boxes = []
            bear_boxes = []

            # Iterate through the DataFrame to collect identified pivot candles as potential blocks
            # We use .iterrows() or index access; .iloc is generally faster for row access by integer position
            for i in range(len(df_copy)):
                # Only consider candles where pivot flags are set (1.0 indicates True)
                # Ensure volume_norm is not None before using it
                current_volume_norm = df_copy['volume_norm'].iloc[i]
                if pd.isna(current_volume_norm): continue # Skip if volume data is incomplete

                # Bearish block candidate: Pivot High identified
                if df_copy['pivot_high'].iloc[i] == 1.0:
                    bear_boxes.append({
                        'timestamp': df_copy.index[i], # The timestamp of the pivot candle
                        'top': df_copy['high'].iloc[i],    # High of the pivot candle
                        'bottom': df_copy['low'].iloc[i],  # Low of the pivot candle
                        'volume': df_copy['volume'].iloc[i], # Raw volume
                        'volume_norm': current_volume_norm # Normalized volume
                        # Could add other info like close price, etc.
                    })
                # Bullish block candidate: Pivot Low identified
                if df_copy['pivot_low'].iloc[i] == 1.0:
                    bull_boxes.append({
                        'timestamp': df_copy.index[i],
                        'top': df_copy['high'].iloc[i],
                        'bottom': df_copy['low'].iloc[i],
                        'volume': df_copy['volume'].iloc[i],
                        'volume_norm': current_volume_norm
                    })

            logger.debug(f"VOB Calc complete. Found {len(bull_boxes)} bull, {len(bear_boxes)} bear block candidates.")
            # Return the dataframe (potentially useful for debugging/plotting) and the collected blocks
            return df_copy, {'bull_boxes': bull_boxes, 'bear_boxes': bear_boxes}

        except KeyError as e:
            logger.error(f"{Fore.RED}VOB Calc Error: Missing required column in DataFrame: {e}.", exc_info=True)
            return df, None # Return original df, indicate failure
        except Exception as e:
            logger.error(f"{Fore.RED}VOB Calc Error: Unexpected error during calculation: {e}", exc_info=True)
            return df, None # Return original df, indicate failure

    # --- Overridden Strategy Method ---

    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates trading signals based on Volumetric Order Block analysis.
        Looks for interaction of the latest candle with recent VOB zones.
        Returns: {'signal': 'long'/'short'/None, 'sl_price': Decimal/None, 'tp_price': None}
        """
        if df is None or df.empty:
            logger.warning("Signal Gen: Input DataFrame is empty.")
            return {'signal': None}
        if len(df) < self.config.vob_pivot_window + 2: # Need enough data for pivots + current candle
             logger.warning(f"Signal Gen: Not enough data ({len(df)}) for VOB analysis with pivot window ({self.config.vob_pivot_window}).")
             return {'signal': None}

        logger.debug("Generating VOB signals...")

        # Calculate order blocks using the helper method
        df_analyzed, order_blocks = self._calculate_volumetric_order_blocks(df)

        if order_blocks is None:
            logger.warning("Could not calculate order blocks, no signal generated.")
            return {'signal': None}

        signals: Dict[str, Any] = {'signal': None, 'sl_price': None, 'tp_price': None}
        try:
            # Get the last complete candle for checking interaction
            # Note: Index -1 might be the current *incomplete* candle if fetch_ohlcv includes it.
            # Assuming fetch_ohlcv returns completed candles, -1 is the last *closed* candle.
            # If strategy should react to current price within an incomplete candle, this needs adjustment.
            # For now, we check the interaction of the *last closed candle* with blocks formed *before* it.
            last_candle = df_analyzed.iloc[-1]
            last_high = last_candle['high']
            last_low = last_candle['low']
            last_close = last_candle['close'] # Might be useful for confirmation filters

            bull_boxes = order_blocks.get('bull_boxes', [])
            bear_boxes = order_blocks.get('bear_boxes', [])

            # Filter for relevant blocks: only those formed *before* the last candle
            relevant_bear_boxes = [box for box in bear_boxes if box['timestamp'] < last_candle.name]
            relevant_bull_boxes = [box for box in bull_boxes if box['timestamp'] < last_candle.name]

            # Focus on the most recent relevant block of each type
            latest_bear_box = max(relevant_bear_boxes, key=lambda box: box['timestamp']) if relevant_bear_boxes else None
            latest_bull_box = max(relevant_bull_boxes, key=lambda box: box['timestamp']) if relevant_bull_boxes else None

            # --- Check Bearish Blocks for Short Signal ---
            # Condition: Price interacts with the latest bearish block.
            # A common interaction is the last candle's low being below the block top,
            # and high being above or within the block bottom.
            # This suggests price has pushed up into the resistance zone.
            if latest_bear_box:
                bear_box_top = latest_bear_box['top']
                bear_box_bottom = latest_bear_box['bottom']

                # Check for interaction: Last candle's range overlaps with the bear box range
                # This covers cases where price wicks into, closes in, or passes through the box
                overlap_with_bear_box = (last_low <= bear_box_top and last_high >= bear_box_bottom)

                # Optional refinement: Add confirmation, e.g., last close below the box top after interaction
                bear_confirmation = last_close < bear_box_top # Example: closing below the resistance zone

                if overlap_with_bear_box and bear_confirmation: # Apply confirmation filter
                    signals['signal'] = 'short'
                    # Set SL slightly above the bear box top + buffer
                    sl_price = bear_box_top * (Decimal("1") + self.config.vob_sl_buffer_pct)
                    signals['sl_price'] = self._quantize_price(sl_price)
                    reason = f"Last candle interacted with Bearish OB @ {bear_box_bottom:.{self.market_info['precision']['price']}f}-{bear_box_top:.{self.market_info['precision']['price']}f} and closed below OB top."
                    logger.info(f"{Fore.RED}Generated SHORT signal ({reason}). Proposed SL: {signals['sl_price']}")
                    # Short signal found, return immediately
                    return signals

            # --- Check Bullish Blocks for Long Signal ---
            # Condition: Price interacts with the latest bullish block.
            # A common interaction is the last candle's high being above the block bottom,
            # and low being below or within the block top.
            # This suggests price has dropped into the support zone.
            if latest_bull_box:
                bull_box_top = latest_bull_box['top']
                bull_box_bottom = latest_bull_box['bottom']

                # Check for interaction: Last candle's range overlaps with the bull box range
                overlap_with_bull_box = (last_high >= bull_box_bottom and last_low <= bull_box_top)

                # Optional refinement: Add confirmation, e.g., last close above the box bottom after interaction
                bull_confirmation = last_close > bull_box_bottom # Example: closing above the support zone

                if overlap_with_bull_box and bull_confirmation: # Apply confirmation filter
                    signals['signal'] = 'long'
                    # Set SL slightly below the bull box bottom - buffer
                    sl_price = bull_box_bottom * (Decimal("1") - self.config.vob_sl_buffer_pct)
                    signals['sl_price'] = self._quantize_price(sl_price)
                    reason = f"Last candle interacted with Bullish OB @ {bull_box_bottom:.{self.market_info['precision']['price']}f}-{bull_box_top:.{self.market_info['precision']['price']}f} and closed above OB bottom."
                    logger.info(f"{Fore.GREEN}Generated LONG signal ({reason}). Proposed SL: {signals['sl_price']}")
                    # Long signal found, return immediately
                    return signals

            # If neither signal condition was met after checking the latest relevant blocks
            logger.debug("No VOB signal generated based on latest relevant block interaction.")
            return signals # Returns {'signal': None, 'sl_price': None, 'tp_price': None}

        except KeyError as e:
            logger.error(f"{Fore.RED}Error generating VOB signals: Missing expected data key {e}.", exc_info=True)
            return {'signal': None}
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error generating VOB signals: {e}", exc_info=True)
            return {'signal': None}


    # --- State Handling Methods ---
    # These methods manage the bot's lifecycle: searching for trades,
    # entering positions, managing open positions, and closing positions.
    # They interact with the core methods and the strategy's signal generation.

    def _handle_searching(self):
        """Handles the STATE_SEARCHING state: fetches data, runs strategy, attempts entry."""
        logger.debug(f"State: {self.state}")
        df = self._fetch_ohlcv()
        if df is None:
            logger.warning("Failed to fetch OHLCV data. Cannot generate signals.")
            return # Stay in SEARCHING, retry on next loop

        # Generate signals using the VOB strategy
        signal_data = self._generate_signals(df)
        signal = signal_data.get('signal')
        strategy_sl_price = signal_data.get('sl_price') # Get SL directly from VOB strategy

        if signal in ['long', 'short']:
            logger.info(f"{Fore.CYAN+Style.BRIGHT}Strategy signal detected: {signal.upper()}!")
            side = 'buy' if signal == 'long' else 'sell'

            # The VOB strategy provides the Stop Loss price
            stop_loss_price = strategy_sl_price
            if stop_loss_price is None:
                 logger.error(f"{Fore.RED}VOB Strategy generated a {signal} signal but did NOT provide an SL price. Cannot proceed with entry.")
                 # This indicates an issue with the strategy implementation, log and abort entry attempt
                 return # Stay in SEARCHING

            # Estimate entry price for position sizing calculation
            # For market orders, the last close is a reasonable estimate for sizing purposes.
            # For limit orders, the limit price itself would be the estimate.
            # Assuming market orders for simplicity based on config default:
            if self.config.entry_order_type == 'market':
                 estimated_entry_price = df['close'].iloc[-1]
            else:
                 # For limit orders, _generate_signals would ideally provide an entry price target.
                 # The current VOB strategy doesn't, so we might use last close or require a strategy update.
                 # For now, we'll assume market entry or that strategy_data includes 'entry_price' for limit.
                 # If using limit orders with VOB, _generate_signals would need to be enhanced.
                 logger.warning(f"{Fore.YELLOW}Entry order type is '{self.config.entry_order_type}'. VOB strategy currently assumes market entry for sizing estimate. Using last close.")
                 estimated_entry_price = df['close'].iloc[-1]
                 # Alternatively, add 'entry_price' to signal_data dict if strategy supports limit entry points.
                 # estimated_entry_price = signal_data.get('entry_price')
                 # if estimated_entry_price is None: ... handle error ...

            if estimated_entry_price is None or estimated_entry_price <= Decimal('0'):
                 logger.error(f"{Fore.RED}Could not determine valid estimated entry price. Cannot proceed with sizing.")
                 return # Stay in SEARCHING

            # Calculate Position Size using the strategy-provided SL
            position_size = self._calculate_position_size(estimated_entry_price, stop_loss_price)

            if position_size is None or position_size <= Decimal("0"):
                 logger.warning(f"{Fore.YELLOW}Could not calculate a valid position size. Entry aborted.")
                 return # Stay in SEARCHING

            # --- Place Entry Order ---
            self.state = STATE_ENTERING # Transition to entering state
            logger.info(f"{Fore.CYAN}Transitioning to {self.state} to place entry order...")

            # Determine actual price for the order if it's a limit order.
            # Since VOB signal only gives SL, assuming MARKET entry or strategy needs enhancement for limit.
            order_price = None # For market order, price is None

            entry_order = self._place_order(
                side=side,
                amount=position_size,
                order_type=self.config.entry_order_type,
                price=order_price # Only needed for limit orders
            )

            if entry_order and entry_order.get('id'):
                order_id = entry_order['id']
                # Target status for confirmation: 'closed' for market, 'open' for limit
                target_status = 'closed' if self.config.entry_order_type == 'market' else 'open'

                if self._confirm_order_status(order_id, target_status=target_status):
                     # Get the actual average fill price for the position
                     # For market orders, fetch_order after status='closed' should give 'average'
                     # For limit orders that just became 'open', this might return None initially.
                     actual_entry_price = self._get_actual_entry_price(order_id)

                     if actual_entry_price is None:
                          # If fetching actual fill price failed (e.g., dry run, or exchange lag),
                          # use the estimated price used for sizing as a fallback.
                          logger.warning(f"{Fore.YELLOW}Could not get actual fill price from exchange for order {order_id}. Using estimated price {estimated_entry_price:.{self.market_info['precision']['price']}f} as fallback.")
                          actual_entry_price = estimated_entry_price
                          # Check if fallback is still invalid
                          if actual_entry_price is None or actual_entry_price <= Decimal('0'):
                              logger.error(f"{Fore.RED}Fallback estimated price is also invalid. Cannot determine entry price. Aborting position tracking.")
                              self.state = STATE_SEARCHING # Go back to searching
                              return

                     # Now that we have the actual or estimated entry price, finalize SL/TP
                     # The initial SL comes directly from the VOB strategy calculation.
                     final_sl = stop_loss_price # This is the VOB-calculated SL price

                     # Calculate Take Profit based on fixed percentage from config, relative to actual entry price
                     final_tp = None
                     if self.config.take_profit_pct and actual_entry_price > Decimal('0'):
                         tp_factor = (Decimal("1") + self.config.take_profit_pct) if signal == 'long' else (Decimal("1") - self.config.take_profit_pct)
                         final_tp = self._quantize_price(actual_entry_price * tp_factor)
                         # Basic sanity check for TP vs SL vs Entry
                         if signal == 'long' and (final_tp <= actual_entry_price or final_tp <= final_sl):
                             logger.warning(f"{Fore.YELLOW}Calculated Long TP ({final_tp}) is not above entry ({actual_entry_price}) or SL ({final_sl}). Disabling TP.")
                             final_tp = None
                         elif signal == 'short' and (final_tp >= actual_entry_price or final_tp >= final_sl):
                             logger.warning(f"{Fore.YELLOW}Calculated Short TP ({final_tp}) is not below entry ({actual_entry_price}) or SL ({final_sl}). Disabling TP.")
                             final_tp = None


                     # Store active position details
                     self.active_position = {
                         'side': signal, # 'long' or 'short'
                         'entry_price': actual_entry_price,
                         'size': position_size, # The quantized amount placed
                         'initial_sl': final_sl, # The strategy-determined SL
                         'current_tsl': final_sl, # Trailing stop starts at the initial SL
                         'take_profit': final_tp, # Calculated TP (can be None)
                         'entry_order_id': order_id,
                         'entry_time': pd.Timestamp.now(tz='UTC') # Record entry time
                     }

                     # Transition to the managing state
                     self.state = STATE_MANAGING_LONG if signal == 'long' else STATE_MANAGING_SHORT
                     logger.info(f"{Fore.GREEN+Style.BRIGHT}Entry successful! Position opened. Transitioning to {self.state}")
                     logger.info(f"  Side: {self.active_position['side'].upper()}")
                     logger.info(f"  Size: {self.active_position['size']} {self.base_currency if self.config.market_type == 'linear' else 'Contracts'}")
                     logger.info(f"  Entry Price: {self.active_position['entry_price']:.{self.market_info['precision']['price']}f}")
                     logger.info(f"  Initial SL: {self.active_position['initial_sl']:.{self.market_info['precision']['price']}f}")
                     if self.active_position['take_profit']:
                          logger.info(f"  Take Profit: {self.active_position['take_profit']:.{self.market_info['precision']['price']}f}")
                     logger.info(f"  Entry Time: {self.active_position['entry_time']}")

                else:
                     logger.error(f"{Fore.RED}Entry order {order_id} failed confirmation or did not reach target status. Aborting entry.")
                     # If a limit order failed to open, it might still be active on the exchange.
                     # A more robust bot might cancel it here. For simplicity, we just log and reset state.
                     self.state = STATE_SEARCHING # Go back to searching

            else:
                logger.error(f"{Fore.RED}Failed to place entry order. Order object is None or missing ID.")
                self.state = STATE_SEARCHING # Go back to searching


    def _handle_managing_position(self):
        """Handles STATE_MANAGING_LONG/SHORT: monitors price, updates TSL, checks exits."""
        logger.debug(f"State: {self.state}")
        if not self.active_position:
             logger.error(f"{Fore.RED}Attempted to manage position, but no active position data found! This is unexpected. Returning to SEARCHING.")
             self.state = STATE_SEARCHING
             return

        current_price = self._fetch_current_price()
        if current_price is None:
             logger.warning("Failed to fetch current price during position management. Retrying on next loop.")
             return # Stay in managing state, retry price fetch

        pos = self.active_position
        side = pos['side']
        entry_price = pos['entry_price']
        current_tsl = pos['current_tsl']
        take_profit = pos['take_profit']

        # Calculate current PnL percentage for logging
        pnl_pct = Decimal("0.0")
        if entry_price > Decimal("0"): # Avoid division by zero
            pnl_pct = ((current_price / entry_price) - Decimal("1")) if side == 'long' else ((entry_price / current_price) - Decimal("1"))

        pnl_color = Fore.GREEN if pnl_pct >= 0 else Fore.RED
        tp_str = f"{take_profit:.{self.market_info['precision']['price']}f}" if take_profit is not None else 'N/A'
        logger.info(f"Managing {side.upper()} | Price={current_price:.{self.market_info['precision']['price']}f} "
                    f"| Entry={entry_price:.{self.market_info['precision']['price']}f} "
                    f"| PnL={pnl_color}{pnl_pct:+.2%}{Style.RESET_ALL} "
                    f"| TP={tp_str} | SL(eff)={current_tsl:.{self.market_info['precision']['price']}f}")

        # --- Update Trailing Stop Loss (TSL) ---
        # TSL moves in the direction of profit but never retreats.
        # It is only updated if the potential new TSL is better (higher for long, lower for short)
        # than the current effective stop loss.
        tsl_buffer = Decimal("1") - self.config.trailing_stop_pct if side == 'long' else Decimal("1") + self.config.trailing_stop_pct
        potential_tsl = self._quantize_price(current_price * tsl_buffer)

        tsl_updated = False
        if side == 'long':
            # For long positions, TSL should only move up.
            # The TSL should be at least the initial SL set by the strategy.
            effective_tsl_candidate = max(potential_tsl, pos['initial_sl'])
            if effective_tsl_candidate > pos['current_tsl']:
                 logger.info(f"{Fore.GREEN}TSL Updated (LONG): {pos['current_tsl']} -> {effective_tsl_candidate:.{self.market_info['precision']['price']}f}")
                 pos['current_tsl'] = effective_tsl_candidate
                 tsl_updated = True
        elif side == 'short':
            # For short positions, TSL should only move down.
            # The TSL should be at most the initial SL set by the strategy.
            effective_tsl_candidate = min(potential_tsl, pos['initial_sl'])
            if effective_tsl_candidate < pos['current_tsl']:
                 logger.info(f"{Fore.GREEN}TSL Updated (SHORT): {pos['current_tsl']} -> {effective_tsl_candidate:.{self.market_info['precision']['price']}f}")
                 pos['current_tsl'] = effective_tsl_candidate
                 tsl_updated = True

        if tsl_updated:
            # Optional: If exchange supports modifying TSL order without canceling/replacing, do it here.
            # Most simple bots just track TSL internally and use a market exit when triggered.
            pass # Simple bots don't update exchange orders constantly

        # --- Check Exit Conditions ---
        exit_reason = None

        # 1. Check Take Profit
        if take_profit is not None:
            if side == 'long' and current_price >= take_profit:
                exit_reason = "Take Profit"
            elif side == 'short' and current_price <= take_profit:
                exit_reason = "Take Profit"

        # 2. Check Stop Loss (TSL or Initial)
        # Only check SL if TP hasn't already triggered
        if exit_reason is None:
            if side == 'long' and current_price <= pos['current_tsl']:
                exit_reason = "Stop Loss (TSL)" if pos['current_tsl'] != pos['initial_sl'] else "Stop Loss (Initial)"
            elif side == 'short' and current_price >= pos['current_tsl']:
                exit_reason = "Stop Loss (TSL)" if pos['current_tsl'] != pos['initial_sl'] else "Stop Loss (Initial)"

        # If an exit condition is met, transition to the exit state
        if exit_reason:
            logger.info(f"{Fore.YELLOW+Style.BRIGHT}Exit condition met: {exit_reason} at price {current_price:.{self.market_info['precision']['price']}f}")
            self.active_position['exit_reason'] = exit_reason # Store reason
            self.state = STATE_EXIT_TRIGGERED # Signal the transition to closing
            logger.info(f"{Fore.CYAN}Transitioning to {self.state}")

        # If no exit condition, stay in managing state

    def _handle_closing(self):
        """Handles STATE_CLOSING: Places the market closing order and confirms closure."""
        logger.debug(f"State: {self.state}")
        if not self.active_position:
             logger.error(f"{Fore.RED}Attempted to close position, but no active position data found! This is unexpected. Returning to SEARCHING.")
             self.state = STATE_SEARCHING
             return

        pos = self.active_position
        # Determine the side needed to close the position
        close_side = 'sell' if pos['side'] == 'long' else 'buy'
        close_amount = pos['size']

        logger.info(f"{Fore.BLUE+Style.BRIGHT}Attempting to place closing order: {close_side.upper()} {close_amount} {self.config.symbol}...")

        # Use params={'reduceOnly': True} to ensure this order only reduces/closes the position,
        # preventing accidental reverse positions if amount exceeds current position size.
        exit_order = self._place_order(
            side=close_side,
            amount=close_amount,
            order_type=self.config.exit_order_type,
            params={'reduceOnly': True} # Essential safety parameter for closing
        )

        if exit_order and exit_order.get('id'):
            order_id = exit_order['id']
            # For exit orders, we always target 'closed' status
            if self._confirm_order_status(order_id, target_status='closed'):
                 logger.info(f"{Fore.GREEN+Style.BRIGHT}Position successfully closed via order {order_id}. Reason: {pos.get('exit_reason', 'N/A')}")
                 self.state = STATE_CLOSED # Transition to the closed state
            else:
                 logger.error(f"{Fore.RED+Style.BRIGHT}Failed to confirm closure order {order_id} status. Manual check required on the exchange!")
                 self.state = STATE_ERROR # Enter error state, requires manual intervention
        else:
            logger.error(f"{Fore.RED+Style.BRIGHT}Failed to place closing order. Order object is None or missing ID. Manual check required!")
            self.state = STATE_ERROR # Enter error state

        # Clear the active position data regardless of success/failure in this state
        # This prevents trying to manage or close the same (potentially failed) position again.
        # If state is ERROR, manual recovery might be needed. If STATE_CLOSED, we are ready to search again.
        if self.state in [STATE_CLOSED, STATE_ERROR]:
             self.active_position = {}
             if self.state == STATE_CLOSED:
                 logger.info("Position data cleared. Ready to search for new opportunities.")


    # --- Main Execution Loop ---
    def run(self):
        """The main operational cycle of the Unified Trader."""
        logger.info(f"{Fore.YELLOW+Style.BRIGHT}=== {APP_NAME} v1.1 Activated ===")
        if self.config.dry_run:
             logger.warning(f"{Back.YELLOW+Fore.BLACK+Style.BRIGHT} DRY RUN MODE ENABLED {Style.RESET_ALL} - No real trades will be executed.")
        logger.info(f"Initial state: {self.state}")

        while self.state not in [STATE_ERROR, STATE_HALTED]:
            cycle_start_time = time.monotonic()
            current_state = self.state # Capture state at the start of the cycle

            try:
                # State Machine Logic
                if current_state == STATE_SEARCHING:
                    self._handle_searching()
                    # State might change to ENTERING or remain SEARCHING

                elif current_state == STATE_ENTERING:
                     # _handle_searching transitions to ENTERING and then immediately attempts placement
                     # If placement fails or confirms, _handle_searching already transitions away from ENTERING.
                     # So, if we somehow land *back* in ENTERING at the start of a loop cycle,
                     # it suggests an issue or a very brief state change.
                     # For this simple bot, just log and maybe revert.
                     logger.warning(f"Unexpectedly in {STATE_ENTERING} state at start of cycle. Reverting to SEARCHING.")
                     self.state = STATE_SEARCHING

                elif current_state in [STATE_MANAGING_LONG, STATE_MANAGING_SHORT]:
                    self._handle_managing_position()
                    # State might change to EXIT_TRIGGERED or remain MANAGING_LONG/SHORT

                elif current_state == STATE_EXIT_TRIGGERED:
                    # Immediately transition to closing state and handle it
                    logger.info(f"State: {self.state} -> Transitioning to {STATE_CLOSING}")
                    self.state = STATE_CLOSING
                    self._handle_closing()
                    # _handle_closing transitions to CLOSED or ERROR

                elif current_state == STATE_CLOSING:
                    # Similar to ENTERING, if we start a loop in CLOSING, it's unusual.
                    # _handle_closing should transition away in one go.
                    logger.warning(f"Unexpectedly in {STATE_CLOSING} state at start of cycle. Attempting close again.")
                    self._handle_closing() # Attempt closing again

                elif current_state == STATE_CLOSED:
                    logger.info(f"State: {self.state} - Position successfully closed. Returning to search mode.")
                    self.state = STATE_SEARCHING # Transition back to searching

                # If state is ERROR or HALTED, the loop condition will catch it.

            except KeyboardInterrupt:
                # Allow clean shutdown via Ctrl+C
                logger.info(f"\n{Fore.YELLOW+Style.BRIGHT}Shutdown signal received. Halting familiar operations...")
                self.state = STATE_HALTED
                break # Exit the loop

            except Exception as e:
                # Catch any unexpected errors in the state handlers
                logger.critical(f"{Fore.RED+Style.BRIGHT}Critical unexpected error in main loop state handler ({current_state}): {e}", exc_info=True)
                self.state = STATE_ERROR # Transition to error state

            # --- Loop Timing ---
            elapsed_time = time.monotonic() - cycle_start_time
            # Sleep duration depends on the current state
            if self.state in [STATE_MANAGING_LONG, STATE_MANAGING_SHORT]:
                # Check more frequently when managing an open position
                base_interval = self.config.manage_interval_seconds
            else:
                # Search/Idle state uses a longer interval
                base_interval = self.config.loop_sleep_seconds

            sleep_duration = max(0.1, base_interval - elapsed_time) # Ensure minimum sleep

            # Only sleep if not in a final state (ERROR/HALTED)
            if self.state not in [STATE_ERROR, STATE_HALTED]:
                 logger.debug(f"State: {self.state} | Cycle Time: {elapsed_time:.2f}s | Sleeping for: {sleep_duration:.2f}s")
                 time.sleep(sleep_duration)

        logger.info(f"{Fore.MAGENTA+Style.BRIGHT}=== {APP_NAME} Deactivated (Final State: {self.state}) ===")


# =============================================================================
# Spell Invocation
# =============================================================================
if __name__ == '__main__':
    print(Fore.MAGENTA + Style.BRIGHT + f"Initializing Pyrmethus's {APP_NAME} v1.1...")
    print(Fore.CYAN + Style.DIM + "Ensure Termux or your environment has network access and required dependencies.")
    print(Fore.RED + Style.BRIGHT + "Trading involves significant risk of loss. Review configuration carefully and test thoroughly with DRY_RUN=True before using real funds.")

    trader = None
    try:
        # 1. Load Configuration
        config = TraderConfig()
        # 2. Initialize Trader with Config
        trader = UnifiedTrader(config)
        # 3. Start the Trading Loop
        trader.run()

    except (SystemExit, ValueError) as e:
        # Specific exceptions caught during configuration or initialization
        logger.critical(f"{Fore.RED + Style.BRIGHT}Initialization or Configuration failed: {e}")
    except KeyboardInterrupt:
        # Handle Ctrl+C during initialization phases before the main loop starts
        logger.info(f"\n{Fore.YELLOW+Style.BRIGHT}Familiar initialization interrupted by user.")
    except Exception as e:
        # Catch any other unexpected errors during setup
        logger.critical(f"{Fore.RED + Style.BRIGHT}Critical setup error: {e}", exc_info=True)

    # Final message after the program exits the main loop or fails initialization
    logger.info(Fore.MAGENTA + "Incantation complete. The terminal rests.")
```

**To Invoke This VOB-Powered Familiar:**

1.  **Save the Code**: Save the complete code block above as a Python file (e.g., `unified_trader_vob.py`).
2.  **Dependencies**: Ensure you have the necessary libraries installed. If not, run:
    ```bash
    pip install ccxt python-dotenv pandas colorama
    ```
3.  **Create/Update `.env` File**: In the same directory as your script, create or update a `.env` file. Add or modify the following lines, filling in your Bybit API details and **crucially, the new VOB parameters**.
    ```dotenv
    # .env file for Unified Trader v1.1 (VOB Strategy)

    # --- Exchange Credentials (Bybit) ---
    # Required for live trading (DRY_RUN=False)
    BYBIT_API_KEY=YOUR_API_KEY_HERE
    BYBIT_API_SECRET=YOUR_SECRET_HERE

    # --- Exchange & Symbol ---
    SYMBOL=BTC/USDT:USDT      # Trading pair (e.g., BTC/USDT:USDT for Bybit USDT perpetual)
    MARKET_TYPE=linear        # 'linear' (USDT/USD settled) or 'inverse' (crypto settled)

    # --- Strategy & Timing ---
    INTERVAL=5m               # Candlestick timeframe (e.g., '1m', '5m', '15m', '1h')
    OHLCV_LIMIT=200           # Number of historical candles to fetch for analysis
    LOOP_SLEEP_SECONDS=30     # Time to wait between cycles when searching for trades
    MANAGE_INTERVAL_SECONDS=10 # Time to wait between cycles when managing an open position

    # --- VOB Strategy Parameters ---
    VOB_VOLUME_EMA_SPAN=20    # EMA period for volume normalization in VOB calculation
    VOB_PIVOT_WINDOW=11       # Window size for identifying pivot points (odd number recommended)
    VOB_SL_BUFFER_PCT=0.001   # Percentage buffer added to the VOB edge for Stop Loss (e.g., 0.001 = 0.1%)

    # --- Risk & Sizing ---
    LEVERAGE=10               # Desired leverage (set on exchange, bot attempts to set)
    RISK_PERCENTAGE=0.005     # Percentage of AVAILABLE BALANCE to risk per trade (e.g., 0.005 = 0.5%)

    # --- Position Management (Exit) ---
    TRAILING_STOP_PCT=0.02    # Percentage for Trailing Stop Loss (e.g., 0.02 = 2%)
    TAKE_PROFIT_PCT=0.04      # Optional fixed Take Profit percentage (e.g., 0.04 = 4%). Comment out or leave empty for no fixed TP.

    # --- Execution ---
    ENTRY_ORDER_TYPE=market   # 'market' or 'limit' (VOB strategy currently works best with market entry)
    EXIT_ORDER_TYPE=market    # 'market' recommended for reliable exits
    ORDER_CHECK_DELAY_SECONDS=2 # Delay between checking order status
    ORDER_CHECK_ATTEMPTS=5    # Number of times to retry order status check
    MAX_FETCH_RETRIES=3       # Number of times to retry fetching data (OHLCV, ticker)

    # --- Operational ---
    DRY_RUN=True              # CRITICAL: Set to True for simulation, False for live trading
    LOG_LEVEL=INFO            # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # LOG_FILE_PATH=unified_trader_vob.log # Optional: Uncomment to log to a file
    ```
4.  **Run the Familiar**: Execute the script from your terminal:
    ```bash
    python unified_trader_vob.py
    