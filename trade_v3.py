"""Scalping Bot v2 - Enhanced Version.

This script implements a cryptocurrency scalping bot using the ccxt library.
It connects to an exchange (default: Bybit Futures), fetches market data,
calculates various technical indicators, determines trade signals based on a scoring system,
places orders, and manages open positions with stop-loss, take-profit, and time-based exits.
"""

import logging
import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# --- Constants ---
CONFIG_FILE_NAME = "config.yaml"
LOG_FILE_NAME = "scalping_bot_v2.log"
DEFAULT_EXCHANGE_ID = "bybit"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_RETRY_MAX = 3
DEFAULT_RETRY_DELAY = 1
DEFAULT_SLEEP_INTERVAL_SECONDS = 10
STRONG_SIGNAL_THRESHOLD_ABS = 3  # Absolute value for strong buy/sell signal score
ENTRY_SIGNAL_THRESHOLD_ABS = 2  # Absolute value for standard buy/sell signal score

# --- Logger Setup ---
# Configure a logger for detailed operational insights
logger = logging.getLogger("ScalpingBot")
logger.setLevel(logging.DEBUG)  # Default level, can be overridden by config

# Formatter for log messages
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console Handler (prints logs to the terminal)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File Handler (writes logs to a file)
try:
    file_handler = logging.FileHandler(LOG_FILE_NAME, mode='a')  # Append mode
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except OSError as e:
    logger.error(f"{Fore.RED}Failed to open log file {LOG_FILE_NAME}: {e}{Style.RESET_ALL}")
    # Continue without file logging if it fails

# Load environment variables from .env file
load_dotenv()


# --- API Retry Decorator ---
def retry_api_call(max_retries: int = DEFAULT_RETRY_MAX, initial_delay: int = DEFAULT_RETRY_DELAY) -> Callable:
    """Decorator to automatically retry API calls with exponential backoff
    upon encountering specific ccxt network or rate limit errors.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any | None:
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded:
                    logger.warning(
                        f"{Fore.YELLOW}Rate limit exceeded for {func.__name__}. Retrying in {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                except ccxt.NetworkError as e:
                    logger.error(
                        f"{Fore.RED}Network error during {func.__name__}: {e}. Retrying in {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                except ccxt.ExchangeError as e:
                    logger.error(
                        f"{Fore.RED}Exchange error during {func.__name__}: {e}. "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    # Specific handling for non-critical errors like "Order not found"
                    if "Order does not exist" in str(e) or "Order not found" in str(e):
                        logger.warning(f"{Fore.YELLOW}Specific exchange error: {e}. Returning None.{Style.RESET_ALL}")
                        return None  # Don't retry if the order simply isn't there
                    # Otherwise, retry for other exchange errors
                except Exception as e:
                    # Catch unexpected errors and log them before retrying
                    logger.error(
                        f"{Fore.RED}Unexpected error during {func.__name__}: {e}. Retrying in {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}",
                        exc_info=True  # Include stack trace for unexpected errors
                    )

                # If we are here, an exception occurred and we need to retry
                if retries < max_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                else:
                    # Max retries reached
                    logger.error(
                        f"{Fore.RED}Max retries ({max_retries}) reached for {func.__name__}. Aborting operation.{Style.RESET_ALL}"
                    )
                    return None  # Indicate failure after all retries

            # Should not be reached if max_retries >= 0, but added for safety
            return None
        return wrapper
    return decorator


# --- Scalping Bot Class ---
class ScalpingBot:
    """A cryptocurrency scalping bot designed to execute trades based on
    technical indicators and order book analysis.
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME) -> None:
        """Initializes the ScalpingBot instance.

        Args:
            config_file: Path to the configuration YAML file.
        """
        logger.info(f"{Fore.CYAN}Initializing Scalping Bot...{Style.RESET_ALL}")
        self.config: dict[str, Any] = {}
        self.load_config(config_file)
        self.validate_config()

        # --- Set Attributes from Config/Env ---
        self.api_key: str | None = os.getenv("BYBIT_API_KEY")
        self.api_secret: str | None = os.getenv("BYBIT_API_SECRET")
        self.exchange_id: str = self.config["exchange"]["exchange_id"]
        self.symbol: str = self.config["trading"]["symbol"]
        self.simulation_mode: bool = self.config["trading"]["simulation_mode"]
        self.entry_order_type: str = self.config["trading"]["entry_order_type"]
        # Renamed for clarity: this is a percentage offset from the current price
        self.limit_order_entry_offset_pct_buy: float = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_entry_offset_pct_sell: float = self.config["trading"]["limit_order_offset_sell"]

        self.order_book_depth: int = self.config["order_book"]["depth"]
        self.imbalance_threshold: float = self.config["order_book"]["imbalance_threshold"]

        # Indicator parameters
        self.volatility_window: int = self.config["indicators"]["volatility_window"]
        self.volatility_multiplier: float = self.config["indicators"]["volatility_multiplier"]
        self.ema_period: int = self.config["indicators"]["ema_period"]
        self.rsi_period: int = self.config["indicators"]["rsi_period"]
        self.macd_short_period: int = self.config["indicators"]["macd_short_period"]
        self.macd_long_period: int = self.config["indicators"]["macd_long_period"]
        self.macd_signal_period: int = self.config["indicators"]["macd_signal_period"]
        self.stoch_rsi_period: int = self.config["indicators"]["stoch_rsi_period"]
        self.stoch_rsi_k_period: int = self.config["indicators"].get("stoch_rsi_k_period", 3)  # Add default if missing
        self.stoch_rsi_d_period: int = self.config["indicators"].get("stoch_rsi_d_period", 3)  # Add default if missing

        # Risk Management parameters
        self.base_stop_loss_pct: float = self.config["risk_management"]["stop_loss_percentage"]
        self.base_take_profit_pct: float = self.config["risk_management"]["take_profit_percentage"]
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.time_based_exit_minutes: int = self.config["risk_management"]["time_based_exit_minutes"]
        self.trailing_stop_loss_percentage: float = self.config["risk_management"]["trailing_stop_loss_percentage"]
        self.order_size_percentage: float = self.config["risk_management"]["order_size_percentage"]
        # Dynamic adjustment factors for TP/SL based on signal confidence
        self.strong_signal_adjustment_factor: float = self.config["risk_management"].get("strong_signal_adjustment_factor", 1.1)
        self.weak_signal_adjustment_factor: float = self.config["risk_management"].get("weak_signal_adjustment_factor", 0.9)

        # --- Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0  # Placeholder for potential future PnL tracking
        self.open_positions: list[dict[str, Any]] = []

        # --- Configure Logging Level ---
        if "logging_level" in self.config:
            log_level_str = self.config["logging_level"].upper()
            log_level = getattr(logging, log_level_str, None)
            if log_level:
                logger.setLevel(log_level)
                # Also set level for handlers if needed (often inherits from logger)
                # console_handler.setLevel(log_level)
                # if file_handler: file_handler.setLevel(log_level)
                logger.info(f"Logging level set to {log_level_str} from config.")
            else:
                logger.warning(
                    f"{Fore.YELLOW}Invalid logging level '{self.config['logging_level']}' in config. "
                    f"Using default ({logging.getLevelName(logger.level)}).{Style.RESET_ALL}"
                )
        else:
             logger.info(f"Using default logging level ({logging.getLevelName(logger.level)}).")

        # --- Initialize Exchange ---
        self.exchange: ccxt.Exchange = self._initialize_exchange()

        if self.simulation_mode:
            logger.warning(f"{Fore.YELLOW}--- RUNNING IN SIMULATION MODE ---{Style.RESET_ALL}")
        else:
             logger.warning(f"{Fore.GREEN}--- RUNNING IN LIVE TRADING MODE ---{Style.RESET_ALL}")

        logger.info(f"{Fore.CYAN}Scalping Bot initialization complete.{Style.RESET_ALL}")

    def load_config(self, config_file: str) -> None:
        """Loads configuration settings from a YAML file."""
        try:
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
            if not self.config:  # Handle empty config file
                 logger.error(f"{Fore.RED}Configuration file {config_file} is empty. Exiting.{Style.RESET_ALL}")
                 sys.exit(1)
            logger.info(f"{Fore.GREEN}Configuration loaded successfully from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.error(f"{Fore.RED}Configuration file '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
                logger.info(f"{Fore.YELLOW}Created a default configuration file: '{config_file}'. "
                            f"Please review and modify it, then restart the bot.{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to create default config file: {e}{Style.RESET_ALL}")
            sys.exit(1)  # Exit after creating default or failing
        except yaml.YAMLError as e:
            logger.error(f"{Fore.RED}Error parsing configuration file {config_file}: {e}. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"{Fore.RED}An unexpected error occurred while loading config: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file with placeholders."""
        default_config = {
            "logging_level": "INFO",  # Default to INFO, DEBUG can be very verbose
            "exchange": {
                "exchange_id": os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE_ID),
            },
            "trading": {
                # --- IMPORTANT: Set your trading symbol ---
                "symbol": "BTC/USDT:USDT",  # Example for Bybit USDT Perpetual
                "simulation_mode": os.getenv("SIMULATION_MODE", "True").lower() in ("true", "1", "yes"),
                "entry_order_type": os.getenv("ENTRY_ORDER_TYPE", "limit").lower(),  # 'limit' or 'market'
                 # Percentage offset from market price for placing limit orders (e.g., 0.001 = 0.1%)
                "limit_order_offset_buy": float(os.getenv("LIMIT_ORDER_OFFSET_BUY", 0.001)),  # Price * (1 - offset)
                "limit_order_offset_sell": float(os.getenv("LIMIT_ORDER_OFFSET_SELL", 0.001)),  # Price * (1 + offset)
            },
            "order_book": {
                "depth": int(os.getenv("ORDER_BOOK_DEPTH", 10)),  # Number of bids/asks levels to fetch
                "imbalance_threshold": float(os.getenv("IMBALANCE_THRESHOLD", 1.5)),  # AskVol/BidVol threshold
            },
            "indicators": {
                "volatility_window": int(os.getenv("VOLATILITY_WINDOW", 20)),  # Lookback period for volatility calc
                "volatility_multiplier": float(os.getenv("VOLATILITY_MULTIPLIER", 0.01)),  # Influence of volatility on order size
                "ema_period": int(os.getenv("EMA_PERIOD", 10)),
                "rsi_period": int(os.getenv("RSI_PERIOD", 14)),
                "macd_short_period": int(os.getenv("MACD_SHORT_PERIOD", 12)),
                "macd_long_period": int(os.getenv("MACD_LONG_PERIOD", 26)),
                "macd_signal_period": int(os.getenv("MACD_SIGNAL_PERIOD", 9)),
                "stoch_rsi_period": int(os.getenv("STOCH_RSI_PERIOD", 14)),
                "stoch_rsi_k_period": 3,  # Smoothing for Stoch RSI %K
                "stoch_rsi_d_period": 3,  # Smoothing for Stoch RSI %D
            },
            "risk_management": {
                 # Percentage of free balance to use for each order
                "order_size_percentage": float(os.getenv("ORDER_SIZE_PERCENTAGE", 0.01)),  # e.g., 0.01 = 1% of balance
                 # Base stop loss percentage from entry price
                "stop_loss_percentage": float(os.getenv("STOP_LOSS_PERCENTAGE", 0.01)),  # e.g., 0.01 = 1%
                 # Base take profit percentage from entry price
                "take_profit_percentage": float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.02)),  # e.g., 0.02 = 2%
                 # Trailing stop loss activation and trail percentage
                "trailing_stop_loss_percentage": float(os.getenv("TRAILING_STOP_LOSS_PERCENTAGE", 0.005)),  # e.g., 0.005 = 0.5%
                "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", 1)),
                 # Max duration for a position before forced exit (in minutes)
                "time_based_exit_minutes": int(os.getenv("TIME_BASED_EXIT_MINUTES", 60)),
                 # Adjustment factors for SL/TP based on signal strength (optional)
                "strong_signal_adjustment_factor": 1.1,  # Increase TP/SL range slightly for strong signals
                "weak_signal_adjustment_factor": 0.9,   # Decrease TP/SL range slightly for weaker signals
            },
        }
        try:
            with open(config_file, "w") as f:
                yaml.dump(default_config, f, indent=4, sort_keys=False)
            logger.info(f"Default configuration file '{config_file}' created.")
        except OSError as e:
            logger.error(f"{Fore.RED}Could not write default config file {config_file}: {e}{Style.RESET_ALL}")
            raise  # Re-raise the exception

    def validate_config(self) -> None:
        """Validates the loaded configuration dictionary."""
        logger.debug("Validating configuration...")
        try:
            # Validate top-level sections
            required_sections = ["exchange", "trading", "order_book", "indicators", "risk_management"]
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required section '{section}' in config file.")

            # Validate specific parameters (add more checks as needed)
            # Trading section
            trading_cfg = self.config["trading"]
            if not isinstance(trading_cfg.get("symbol"), str) or not trading_cfg["symbol"]:
                raise ValueError("'trading.symbol' must be a non-empty string.")
            if not isinstance(trading_cfg.get("simulation_mode"), bool):
                raise ValueError("'trading.simulation_mode' must be a boolean (true/false).")
            if trading_cfg.get("entry_order_type") not in ["market", "limit"]:
                raise ValueError("'trading.entry_order_type' must be 'market' or 'limit'.")
            if not isinstance(trading_cfg.get("limit_order_offset_buy"), (int, float)) or trading_cfg["limit_order_offset_buy"] < 0:
                raise ValueError("'trading.limit_order_offset_buy' must be a non-negative number.")
            if not isinstance(trading_cfg.get("limit_order_offset_sell"), (int, float)) or trading_cfg["limit_order_offset_sell"] < 0:
                raise ValueError("'trading.limit_order_offset_sell' must be a non-negative number.")

            # Order Book section
            order_book_cfg = self.config["order_book"]
            if not isinstance(order_book_cfg.get("depth"), int) or order_book_cfg["depth"] <= 0:
                raise ValueError("'order_book.depth' must be a positive integer.")
            if not isinstance(order_book_cfg.get("imbalance_threshold"), (int, float)) or order_book_cfg["imbalance_threshold"] <= 0:
                 raise ValueError("'order_book.imbalance_threshold' must be a positive number.")

            # Indicators section (example validation)
            indicators_cfg = self.config["indicators"]
            for key, type_ in [
                ("volatility_window", int), ("ema_period", int), ("rsi_period", int),
                ("macd_short_period", int), ("macd_long_period", int), ("macd_signal_period", int),
                ("stoch_rsi_period", int), ("stoch_rsi_k_period", int), ("stoch_rsi_d_period", int),
                ("volatility_multiplier", (int, float))
            ]:
                 val = indicators_cfg.get(key)
                 if val is None:
                      raise ValueError(f"Missing 'indicators.{key}' in config.")
                 if not isinstance(val, type_) or (isinstance(val, (int, float)) and val < 0):
                     raise ValueError(f"'indicators.{key}' must be a non-negative {type_}.")

            # Risk Management section
            risk_cfg = self.config["risk_management"]
            for key, min_val, max_val in [
                ("order_size_percentage", 0, 1),
                ("stop_loss_percentage", 0, 1),
                ("take_profit_percentage", 0, 1),
                ("trailing_stop_loss_percentage", 0, 1),
                ("strong_signal_adjustment_factor", 0, None),
                ("weak_signal_adjustment_factor", 0, None),
            ]:
                val = risk_cfg.get(key)
                if val is None:
                    raise ValueError(f"Missing 'risk_management.{key}' in config.")
                if not isinstance(val, (int, float)):
                     raise ValueError(f"'risk_management.{key}' must be a number.")
                if val < min_val or (max_val is not None and val > max_val):
                     raise ValueError(f"'risk_management.{key}' must be between {min_val} and {max_val}.")

            for key, min_val in [("max_open_positions", 1), ("time_based_exit_minutes", 1)]:
                 val = risk_cfg.get(key)
                 if val is None:
                     raise ValueError(f"Missing 'risk_management.{key}' in config.")
                 if not isinstance(val, int) or val < min_val:
                     raise ValueError(f"'risk_management.{key}' must be an integer >= {min_val}.")

            logger.info(f"{Fore.GREEN}Configuration validated successfully.{Style.RESET_ALL}")

        except ValueError as e:
            logger.error(f"{Fore.RED}Configuration validation failed: {e}. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"{Fore.RED}An unexpected error occurred during config validation: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and connects to the specified exchange."""
        logger.info(f"Initializing connection to {self.exchange_id.upper()}...")
        if not self.api_key or not self.api_secret:
             if not self.simulation_mode:
                  logger.error(f"{Fore.RED}API Key or Secret not found in environment variables (e.g., BYBIT_API_KEY). "
                               f"Cannot run in live mode. Exiting.{Style.RESET_ALL}")
                  sys.exit(1)
             else:
                  logger.warning(f"{Fore.YELLOW}API Key/Secret not found. Running in simulation mode only.{Style.RESET_ALL}")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': self.api_key if not self.simulation_mode else None,
                'secret': self.api_secret if not self.simulation_mode else None,
                # Common options, adjust as needed per exchange
                'options': {
                    'defaultType': 'future',  # Or 'spot', 'margin', etc.
                    # 'adjustForTimeDifference': True, # Enable if experiencing timestamp issues
                },
                'enableRateLimit': True,  # Enable ccxt's built-in rate limiter
                # 'recvWindow': 10000, # Example: Increase recvWindow if needed (milliseconds)
            })

            # Set sandbox mode if available and in simulation (check exchange docs)
            if self.simulation_mode and self.exchange_id in ['bybit', 'binance']:  # Add other exchanges supporting sandbox
                logger.info("Attempting to enable Sandbox Mode for simulation.")
                exchange.set_sandbox_mode(True)

            # Load markets to ensure connectivity and symbol availability
            exchange.load_markets()
            logger.debug(f"Markets loaded for {self.exchange_id.upper()}.")

            # Verify the trading symbol exists on the exchange
            if self.symbol not in exchange.markets:
                logger.error(f"{Fore.RED}Symbol '{self.symbol}' not found on {self.exchange_id.upper()}. "
                             f"Available symbols: {list(exchange.markets.keys())[:10]}... Exiting.{Style.RESET_ALL}")
                sys.exit(1)

            logger.info(f"{Fore.GREEN}Successfully connected to {self.exchange_id.upper()} "
                        f"({Fore.YELLOW}SIMULATION MODE{Style.RESET_ALL}" if self.simulation_mode else f"{Fore.GREEN}LIVE MODE{Style.RESET_ALL}).")
            return exchange

        except AttributeError:
            logger.error(f"{Fore.RED}Exchange ID '{self.exchange_id}' not found in ccxt library. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        except ccxt.AuthenticationError as e:
            logger.error(f"{Fore.RED}Authentication failed for {self.exchange_id.upper()}: {e}. Check API keys. Exiting.{Style.RESET_ALL}")
            sys.exit(1)
        except ccxt.ExchangeNotAvailable as e:
             logger.error(f"{Fore.RED}Exchange {self.exchange_id.upper()} is not available: {e}. Exiting.{Style.RESET_ALL}")
             sys.exit(1)
        except ccxt.RequestTimeout as e:
             logger.error(f"{Fore.RED}Connection timed out while connecting to {self.exchange_id.upper()}: {e}. Exiting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            logger.error(f"{Fore.RED}Error initializing exchange {self.exchange_id.upper()}: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    # --- Data Fetching Methods ---

    @retry_api_call()
    def fetch_market_price(self) -> float | None:
        """Fetches the last traded price for the trading symbol."""
        logger.debug(f"Fetching market price for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            price = float(ticker['last'])
            logger.debug(f"Fetched market price for {self.symbol}: {price}")
            return price
        else:
            logger.warning(f"{Fore.YELLOW}Could not fetch last market price for {self.symbol}. Ticker data: {ticker}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_order_book(self) -> float | None:
        """Fetches the order book and calculates the volume imbalance ratio
        (Total Ask Volume / Total Bid Volume) within the specified depth.
        """
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if bids and asks:
            # Calculate total volume within the specified depth
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)

            if bid_volume > 0:
                imbalance_ratio = ask_volume / bid_volume
                logger.debug(
                    f"Order Book ({self.symbol}) - Top Bid: {bids[0][0]}, Top Ask: {asks[0][0]}, "
                    f"Bid Vol: {bid_volume:.4f}, Ask Vol: {ask_volume:.4f}, Imbalance: {imbalance_ratio:.3f}"
                )
                return imbalance_ratio
            else:
                # Avoid division by zero if there's no bid volume in the fetched depth
                logger.warning(f"{Fore.YELLOW}No bid volume found in the top {self.order_book_depth} levels for {self.symbol}.{Style.RESET_ALL}")
                return float('inf')  # Return infinity to indicate extreme ask dominance
        else:
            logger.warning(f"{Fore.YELLOW}Order book data incomplete or unavailable for {self.symbol}. Bids: {len(bids)}, Asks: {len(asks)}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_historical_data(self, limit: int | None = None) -> pd.DataFrame | None:
        """Fetches historical OHLCV data and returns it as a Pandas DataFrame.

        Args:
            limit: The number of candles to fetch. Defaults to a value sufficient
                   for the longest indicator calculation.

        Returns:
            A Pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            or None if fetching fails or data is insufficient.
        """
        if limit is None:
             # Calculate required limit based on longest indicator period + buffer
            limit = max(
                self.volatility_window,
                self.ema_period,
                self.rsi_period + 1,  # RSI needs n+1 periods
                self.macd_long_period,
                self.stoch_rsi_period + self.stoch_rsi_k_period + self.stoch_rsi_d_period - 2  # Stoch RSI needs underlying RSI + smoothing
            ) + 5  # Add a small buffer

        logger.debug(f"Fetching {limit} historical OHLCV candles for {self.symbol} ({DEFAULT_TIMEFRAME})...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=DEFAULT_TIMEFRAME, limit=limit)
            if not ohlcv:
                logger.warning(f"{Fore.YELLOW}No historical OHLCV data returned for {self.symbol}.{Style.RESET_ALL}")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
            df.set_index('timestamp', inplace=True)  # Optional: set timestamp as index

             # Convert columns to numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaN values that might have occurred during conversion
            df.dropna(inplace=True)

            if len(df) < limit:
                 logger.warning(
                     f"{Fore.YELLOW}Insufficient historical data returned for {self.symbol}. "
                     f"Fetched {len(df)} candles, needed {limit}. Results may be inaccurate.{Style.RESET_ALL}"
                 )
                 # Decide if you want to proceed with less data or return None
                 # return None # Option: Return None if data is insufficient

            logger.debug(f"Successfully fetched and processed {len(df)} historical candles for {self.symbol}.")
            return df

        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching or processing historical data for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    @retry_api_call()
    def fetch_balance(self, currency_code: str = 'USDT') -> float:
        """Fetches the free balance for a specific currency (default: USDT).

        Args:
            currency_code: The currency symbol (e.g., 'USDT', 'BTC').

        Returns:
            The free balance as a float, or 0.0 if unavailable or error occurs.
        """
        logger.debug(f"Fetching balance for {currency_code}...")
        try:
            balance_data = self.exchange.fetch_balance()
            # Accessing free balance safely using .get()
            free_balance = balance_data.get('free', {}).get(currency_code, 0.0)
            if free_balance is None: free_balance = 0.0  # Handle potential None value
            logger.debug(f"Fetched free balance for {currency_code}: {free_balance}")
            return float(free_balance)
        except Exception as e:
            logger.error(f"{Fore.RED}Could not fetch balance for {currency_code}: {e}{Style.RESET_ALL}", exc_info=True)
            return 0.0  # Return 0 on error to prevent issues downstream

    # --- Indicator Calculation Methods ---

    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> float | None:
        """Calculates historical volatility (standard deviation of log returns)."""
        if close_prices is None or len(close_prices) < window:
            logger.debug(f"Insufficient data for volatility calculation (need {window}, got {len(close_prices)}).")
            return None
        # Use log returns for volatility calculation
        log_returns = np.log(close_prices / close_prices.shift(1))
        # Calculate rolling standard deviation of log returns
        volatility = log_returns.rolling(window=window).std().iloc[-1]
        # Optional: Annualize volatility (depends on timeframe)
        # volatility *= np.sqrt(periods_per_year) # e.g., 365*24*60 for 1m timeframe
        logger.debug(f"Calculated volatility ({window}-period): {volatility:.5f}")
        return volatility

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> float | None:
        """Calculates the Exponential Moving Average (EMA)."""
        if close_prices is None or len(close_prices) < period:
            logger.debug(f"Insufficient data for EMA calculation (need {period}, got {len(close_prices)}).")
            return None
        ema = close_prices.ewm(span=period, adjust=False).mean().iloc[-1]
        logger.debug(f"Calculated EMA ({period}-period): {ema:.4f}")
        return ema

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> float | None:
        """Calculates the Relative Strength Index (RSI)."""
        if close_prices is None or len(close_prices) < period + 1:
            logger.debug(f"Insufficient data for RSI calculation (need {period + 1}, got {len(close_prices)}).")
            return None
        delta = close_prices.diff(1)
        gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1 / period, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        logger.debug(f"Calculated RSI ({period}-period): {rsi_value:.2f}")
        return rsi_value

    @staticmethod
    def calculate_macd(close_prices: pd.Series, short_period: int, long_period: int, signal_period: int) -> tuple[float | None, float | None, float | None]:
        """Calculates Moving Average Convergence Divergence (MACD), Signal line, and Histogram."""
        if close_prices is None or len(close_prices) < long_period + signal_period:
            logger.debug(f"Insufficient data for MACD calculation (need {long_period + signal_period}, got {len(close_prices)}).")
            return None, None, None

        ema_short = close_prices.ewm(span=short_period, adjust=False).mean()
        ema_long = close_prices.ewm(span=long_period, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        hist_val = histogram.iloc[-1]
        logger.debug(f"Calculated MACD ({short_period},{long_period},{signal_period}): MACD={macd_val:.4f}, Signal={signal_val:.4f}, Hist={hist_val:.4f}")
        return macd_val, signal_val, hist_val

    @staticmethod
    def calculate_stoch_rsi(close_prices: pd.Series, rsi_period: int, stoch_period: int, k_period: int, d_period: int) -> tuple[float | None, float | None]:
        """Calculates Stochastic RSI %K and %D lines."""
        min_len = rsi_period + stoch_period + max(k_period, d_period) - 2  # Approximate minimum length
        if close_prices is None or len(close_prices) < min_len:
             logger.debug(f"Insufficient data for Stoch RSI calculation (need ~{min_len}, got {len(close_prices)}).")
             return None, None

        # Calculate RSI first
        delta = close_prices.diff(1)
        gain = delta.where(delta > 0, 0).ewm(alpha=1 / rsi_period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.dropna()  # Drop initial NaNs from RSI calculation

        if len(rsi) < stoch_period:
            logger.debug(f"Insufficient RSI values for Stoch RSI calculation (need {stoch_period}, got {len(rsi)}).")
            return None, None

        # Calculate Stoch RSI
        min_rsi = rsi.rolling(window=stoch_period).min()
        max_rsi = rsi.rolling(window=stoch_period).max()
        stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
        stoch_rsi = stoch_rsi.dropna()  # Drop NaNs from rolling window

        if len(stoch_rsi) < max(k_period, d_period):
             logger.debug(f"Insufficient Stoch RSI values for smoothing (need {max(k_period, d_period)}, got {len(stoch_rsi)}).")
             return None, None

        # Calculate %K and %D (smoothed)
        stoch_k = stoch_rsi.rolling(window=k_period).mean()
        stoch_d = stoch_k.rolling(window=d_period).mean()

        k_val = stoch_k.iloc[-1]
        d_val = stoch_d.iloc[-1]

        if pd.isna(k_val) or pd.isna(d_val):
            logger.debug("Stoch RSI K or D calculation resulted in NaN.")
            return None, None

        logger.debug(f"Calculated Stoch RSI ({rsi_period},{stoch_period},{k_period},{d_period}): %K={k_val:.2f}, %D={d_val:.2f}")
        return k_val, d_val

    # --- Trading Logic Methods ---

    def calculate_order_size(self, base_currency: str = 'USDT') -> float:
        """Calculates the order size in the base currency (e.g., USDT),
        potentially adjusted by volatility.

        Args:
            base_currency: The currency to calculate the size in (usually the quote currency).

        Returns:
            The calculated order size in the base currency, or 0.0 if balance is too low or calculation fails.
        """
        balance = self.fetch_balance(currency_code=base_currency)
        if balance <= 0:
            logger.warning(f"{Fore.YELLOW}Insufficient balance ({balance} {base_currency}) to calculate order size.{Style.RESET_ALL}")
            return 0.0

        # Fetch recent close prices for volatility calculation
        historical_data = self.fetch_historical_data(limit=self.volatility_window + 1)  # Need N+1 for diff
        volatility = None
        if historical_data is not None and not historical_data.empty:
             volatility = self.calculate_volatility(historical_data['close'], self.volatility_window)

        # Base size calculation
        base_order_size = balance * self.order_size_percentage

        # Adjust size based on volatility (optional)
        if volatility is not None and self.volatility_multiplier > 0:
            # Example adjustment: Increase size slightly in low vol, decrease in high vol
            # This logic can be customized heavily.
            # Simple inverse relationship: size_factor = 1 / (1 + volatility * self.volatility_multiplier)
            # Simple direct relationship: size_factor = 1 + volatility * self.volatility_multiplier
            # Let's use the original direct relationship logic for now:
            size_factor = 1 + (volatility * self.volatility_multiplier)
            adjusted_size = base_order_size * size_factor
            logger.info(f"Volatility ({volatility:.5f}) adjusted order size factor: {size_factor:.3f}")
        else:
            logger.info("Calculating order size without volatility adjustment.")
            adjusted_size = base_order_size

        # Ensure size doesn't exceed a max percentage of balance (e.g., 5%)
        # max_allowed_size = balance * 0.05 # Example hard cap
        # final_size = min(adjusted_size, max_allowed_size)
        final_size = adjusted_size  # Use adjusted size directly for now, rely on order_size_percentage

        # Get market info for minimum order size
        market_info = self.exchange.market(self.symbol)
        min_cost = market_info.get('limits', {}).get('cost', {}).get('min')
        market_info.get('limits', {}).get('amount', {}).get('min')

        if min_cost is not None and final_size < min_cost:
             logger.warning(f"{Fore.YELLOW}Calculated order size {final_size:.4f} {base_currency} is below minimum cost {min_cost} {base_currency}. Setting size to 0.{Style.RESET_ALL}")
             return 0.0

        # Note: For amount check, we need the price. This check might be better placed right before ordering.
        # if min_amount is not None: ...

        logger.info(f"{Fore.CYAN}Calculated order size: {final_size:.4f} {base_currency} (Balance: {balance:.2f} {base_currency}){Style.RESET_ALL}")
        return final_size

    def compute_trade_signal_score(self, price: float, indicators: dict[str, float | None], orderbook_imbalance: float | None) -> tuple[int, list[str]]:
        """Computes a trade signal score based on various technical indicators and order book imbalance.
        Positive score suggests LONG, negative suggests SHORT.

        Args:
            price: The current market price.
            indicators: A dictionary containing calculated indicator values (ema, rsi, etc.).
            orderbook_imbalance: The calculated order book imbalance ratio.

        Returns:
            A tuple containing the integer signal score and a list of strings explaining the score components.
        """
        score = 0
        reasons = []

        # 1. Order Book Imbalance
        if orderbook_imbalance is not None:
            if orderbook_imbalance < (1 / self.imbalance_threshold):  # More bid volume
                score += 1
                reasons.append(f"{Fore.GREEN}+1: Order book shows buy pressure (Imb: {orderbook_imbalance:.2f}){Style.RESET_ALL}")
            elif orderbook_imbalance > self.imbalance_threshold:  # More ask volume
                score -= 1
                reasons.append(f"{Fore.RED}-1: Order book shows sell pressure (Imb: {orderbook_imbalance:.2f}){Style.RESET_ALL}")
            else:
                 reasons.append(f"{Fore.WHITE}0: Order book relatively balanced (Imb: {orderbook_imbalance:.2f}){Style.RESET_ALL}")

        # 2. EMA Trend
        ema = indicators.get('ema')
        if ema is not None:
            if price > ema:
                score += 1
                reasons.append(f"{Fore.GREEN}+1: Price ({price:.2f}) above EMA ({ema:.2f}) (bullish trend){Style.RESET_ALL}")
            elif price < ema:
                score -= 1
                reasons.append(f"{Fore.RED}-1: Price ({price:.2f}) below EMA ({ema:.2f}) (bearish trend){Style.RESET_ALL}")
            else:
                 reasons.append(f"{Fore.WHITE}0: Price ({price:.2f}) equals EMA ({ema:.2f}){Style.RESET_ALL}")

        # 3. RSI Momentum/Overbought/Oversold
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < 30:
                score += 1  # Oversold condition, potential reversal up
                reasons.append(f"{Fore.GREEN}+1: RSI ({rsi:.2f}) < 30 (oversold){Style.RESET_ALL}")
            elif rsi > 70:
                score -= 1  # Overbought condition, potential reversal down
                reasons.append(f"{Fore.RED}-1: RSI ({rsi:.2f}) > 70 (overbought){Style.RESET_ALL}")
            else:
                # Optional: Add points based on RSI trend (e.g., RSI rising/falling)
                reasons.append(f"{Fore.WHITE}0: RSI ({rsi:.2f}) is neutral (30-70){Style.RESET_ALL}")

        # 4. MACD Momentum/Cross
        macd_line = indicators.get('macd_line')
        macd_signal = indicators.get('macd_signal')
        # macd_hist = indicators.get('macd_hist') # Histogram can also be used
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:  # MACD line crossed above signal line
                score += 1
                reasons.append(f"{Fore.GREEN}+1: MACD line ({macd_line:.4f}) above signal ({macd_signal:.4f}) (bullish momentum){Style.RESET_ALL}")
            else:  # MACD line crossed below signal line
                score -= 1
                reasons.append(f"{Fore.RED}-1: MACD line ({macd_line:.4f}) below signal ({macd_signal:.4f}) (bearish momentum){Style.RESET_ALL}")

        # 5. Stochastic RSI Overbought/Oversold
        stoch_rsi_k = indicators.get('stoch_k')
        stoch_rsi_d = indicators.get('stoch_d')
        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            # Standard Stoch RSI signals
            if stoch_rsi_k < 20 and stoch_rsi_d < 20:  # Both lines below 20 (oversold)
                score += 1
                reasons.append(f"{Fore.GREEN}+1: Stoch RSI K ({stoch_rsi_k:.2f}) & D ({stoch_rsi_d:.2f}) < 20 (oversold){Style.RESET_ALL}")
            elif stoch_rsi_k > 80 and stoch_rsi_d > 80:  # Both lines above 80 (overbought)
                score -= 1
                reasons.append(f"{Fore.RED}-1: Stoch RSI K ({stoch_rsi_k:.2f}) & D ({stoch_rsi_d:.2f}) > 80 (overbought){Style.RESET_ALL}")
            # Optional: Add signals for K/D crossovers within neutral territory
            # elif stoch_k > stoch_d and stoch_k < 80 and stoch_d < 80: # Bullish crossover
            #     score += 0.5 # Fractional score example
            # elif stoch_k < stoch_d and stoch_k > 20 and stoch_d > 20: # Bearish crossover
            #     score -= 0.5
            else:
                 reasons.append(f"{Fore.WHITE}0: Stoch RSI K ({stoch_rsi_k:.2f}) & D ({stoch_rsi_d:.2f}) neutral{Style.RESET_ALL}")

        logger.debug(f"Signal score calculated: {score}")
        return int(round(score)), reasons  # Ensure integer score

    def place_order(
        self,
        side: str,
        order_size_quote: float,  # Size in quote currency (e.g., USDT)
        confidence_level: int,  # The signal score
        order_type: str = "market",
        limit_price: float | None = None,  # Required for limit orders
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict[str, Any] | None:
        """Places a trade order on the exchange or simulates it.

        Args:
            side: 'buy' or 'sell'.
            order_size_quote: The desired order size in the quote currency (e.g., USDT).
            confidence_level: The signal score associated with this order.
            order_type: 'market' or 'limit'.
            limit_price: The price for a limit order.
            stop_loss_price: The stop-loss trigger price.
            take_profit_price: The take-profit trigger price.

        Returns:
            A dictionary representing the order (simulated or actual) or None if placement fails.
        """
        current_price = self.fetch_market_price()
        if current_price is None:
            logger.error(f"{Fore.RED}Cannot place {side} order: Failed to fetch current market price.{Style.RESET_ALL}")
            return None

        # --- Calculate order size in base currency ---
        order_size_base = order_size_quote / current_price  # Size in BTC for BTC/USDT

        # --- Check minimum order amount ---
        market_info = self.exchange.market(self.symbol)
        min_amount = market_info.get('limits', {}).get('amount', {}).get('min')
        if min_amount is not None and order_size_base < min_amount:
             logger.warning(f"{Fore.YELLOW}Calculated order size {order_size_base:.8f} {market_info['base']} "
                            f"is below minimum amount {min_amount} {market_info['base']}. Cannot place order.{Style.RESET_ALL}")
             return None

        # --- Prepare order parameters ---
        params = {}
        # Note: SL/TP syntax varies significantly between exchanges (ccxt unified params help but aren't perfect)
        # Bybit USDT perpetuals use 'stopLoss'/'takeProfit' in params for createOrder
        if stop_loss_price is not None:
            params['stopLoss'] = self.exchange.price_to_precision(self.symbol, stop_loss_price)
        if take_profit_price is not None:
            params['takeProfit'] = self.exchange.price_to_precision(self.symbol, take_profit_price)

        # Adjust precision for amount and price
        amount_precise = self.exchange.amount_to_precision(self.symbol, order_size_base)
        limit_price_precise = None
        if order_type == "limit":
            if limit_price is None:
                logger.error(f"{Fore.RED}Limit price is required for limit orders.{Style.RESET_ALL}")
                return None
            limit_price_precise = self.exchange.price_to_precision(self.symbol, limit_price)

        # --- Simulation Mode ---
        if self.simulation_mode:
            simulated_order = {
                "id": f"sim_{int(time.time() * 1000)}",  # Unique simulated ID
                "timestamp": int(time.time() * 1000),
                "datetime": pd.to_datetime(time.time(), unit='s').isoformat(),
                "symbol": self.symbol,
                "type": order_type,
                "side": side,
                "amount": float(amount_precise),
                "price": float(limit_price_precise) if order_type == "limit" else float(current_price),  # Use current price for market sim
                "cost": float(amount_precise) * (float(limit_price_precise) if order_type == "limit" else float(current_price)),
                "status": "closed" if order_type == 'market' else 'open',  # Simulate market fill, limit open
                "filled": float(amount_precise) if order_type == 'market' else 0.0,
                "remaining": 0.0 if order_type == 'market' else float(amount_precise),
                "average": float(limit_price_precise) if order_type == "limit" else float(current_price),
                "info": {  # Add custom info
                     "simulated": True,
                     "stopLoss": params.get('stopLoss'),
                     "takeProfit": params.get('takeProfit'),
                     "confidence": confidence_level
                 }
            }
            log_color = Fore.GREEN if side == 'buy' else Fore.RED
            logger.info(
                f"{log_color}[SIMULATION] {'Limit' if order_type == 'limit' else 'Market'} {side.upper()} Order "
                f"Size: {simulated_order['amount']} {market_info['base']}, "
                f"Price: {simulated_order['price']:.2f}{' (Limit)' if order_type == 'limit' else ''}, "
                f"Value: {simulated_order['cost']:.2f} {market_info['quote']}, "
                f"SL: {simulated_order['info']['stopLoss']}, TP: {simulated_order['info']['takeProfit']}, "
                f"Confidence: {confidence_level}{Style.RESET_ALL}"
            )
            return simulated_order

        # --- Live Trading Mode ---
        else:
            try:
                order = None
                log_color = Fore.GREEN if side == 'buy' else Fore.RED
                logger.info(f"{log_color}Attempting to place LIVE {order_type.upper()} {side.upper()} order...{Style.RESET_ALL}")

                if order_type == "market":
                    order = self.exchange.create_market_order(
                        symbol=self.symbol,
                        side=side,
                        amount=float(amount_precise),
                        params=params
                    )
                elif order_type == "limit":
                    order = self.exchange.create_limit_order(
                        symbol=self.symbol,
                        side=side,
                        amount=float(amount_precise),
                        price=float(limit_price_precise),
                        params=params
                    )
                else:
                    logger.error(f"{Fore.RED}Unsupported order type for live trading: {order_type}{Style.RESET_ALL}")
                    return None

                # Log successful order placement
                # Use .get() for optional fields like price which might not be in market order response immediately
                order_price = order.get('price') or (limit_price_precise if order_type == 'limit' else current_price)  # Best guess price
                logger.info(
                    f"{log_color}LIVE Order Placed Successfully: "
                    f"ID: {order['id']}, Type: {order['type']}, Side: {order['side'].upper()}, "
                    f"Amount: {order['amount']}, Price: {order_price}, "
                    f"SL: {params.get('stopLoss', 'N/A')}, TP: {params.get('takeProfit', 'N/A')}, "
                    f"Status: {order['status']}, Confidence: {confidence_level}{Style.RESET_ALL}"
                )
                return order

            # --- Specific Error Handling for Orders ---
            except ccxt.InsufficientFunds as e:
                logger.error(f"{Fore.RED}LIVE Order Failed (Insufficient Funds): Cannot place {side} order for {amount_precise} {market_info['base']}. {e}{Style.RESET_ALL}")
                return None
            except ccxt.InvalidOrder as e:
                logger.error(f"{Fore.RED}LIVE Order Failed (Invalid Order): Parameters likely incorrect. Amount={amount_precise}, Price={limit_price_precise}, SL={stop_loss_price}, TP={take_profit_price}. Error: {e}{Style.RESET_ALL}")
                return None
            except ccxt.OrderNotFound as e:  # Should not happen during creation, but possible in complex scenarios
                logger.error(f"{Fore.RED}LIVE Order Failed (Order Not Found): {e}{Style.RESET_ALL}")
                return None
            except ccxt.ExchangeError as e:  # Catch other specific exchange errors
                 logger.error(f"{Fore.RED}LIVE Order Failed (Exchange Error): {e}{Style.RESET_ALL}", exc_info=True)
                 return None
            except Exception as e:  # Catch any other unexpected error during order placement
                logger.error(f"{Fore.RED}LIVE Order Failed (Unexpected Error): Failed to place {side} order. Error: {e}{Style.RESET_ALL}", exc_info=True)
                return None

    def manage_positions(self) -> None:
        """Manages open positions: checks for SL/TP triggers, time-based exits,
        and activates/updates trailing stop losses.
        """
        if not self.open_positions:
            # logger.debug("No open positions to manage.")
            return

        logger.debug(f"Managing {len(self.open_positions)} open position(s)...")
        current_price = self.fetch_market_price()
        if current_price is None:
            logger.warning(f"{Fore.YELLOW}Cannot manage positions: Failed to fetch current market price.{Style.RESET_ALL}")
            return

        # Iterate over a copy of the list to allow removal during iteration
        for position in list(self.open_positions):
            position_id = position.get("id", "N/A")  # Use order ID if available
            entry_price = position["entry_price"]
            position_side = position["side"]
            order_size = position["size"]
            entry_time = position["entry_time"]
            confidence = position.get("confidence", 0)
            base_sl = position["stop_loss"]
            base_tp = position["take_profit"]
            trailing_sl_active = "trailing_stop_loss" in position
            current_trailing_sl = position.get("trailing_stop_loss")

            exit_reason = None  # Track why we are exiting

            # --- 1. Time-Based Exit Check ---
            time_elapsed_minutes = (time.time() - entry_time) / 60
            if time_elapsed_minutes >= self.time_based_exit_minutes:
                logger.info(
                    f"{Fore.YELLOW}Time-based exit triggered for {position_side.upper()} position "
                    f"(ID: {position_id}, Age: {time_elapsed_minutes:.1f} mins >= {self.time_based_exit_minutes} mins){Style.RESET_ALL}"
                )
                exit_reason = "Time Limit Reached"

            # --- 2. SL/TP Check ---
            effective_stop_loss = current_trailing_sl if trailing_sl_active else base_sl

            if position_side == "buy":
                # Stop Loss Check
                if current_price <= effective_stop_loss:
                    logger.info(
                        f"{Fore.RED}Stop-loss triggered for LONG position (ID: {position_id}). "
                        f"Price ({current_price:.2f}) <= SL ({effective_stop_loss:.2f})"
                        f"{' (Trailing)' if trailing_sl_active else ''}{Style.RESET_ALL}"
                    )
                    exit_reason = "Stop Loss Hit"
                # Take Profit Check
                elif current_price >= base_tp:
                    logger.info(
                        f"{Fore.GREEN}Take-profit triggered for LONG position (ID: {position_id}). "
                        f"Price ({current_price:.2f}) >= TP ({base_tp:.2f}){Style.RESET_ALL}"
                    )
                    exit_reason = "Take Profit Hit"
                # Trailing Stop Activation/Update
                elif self.trailing_stop_loss_percentage > 0:
                     # Activate TSL if price moves significantly in favor (e.g., crosses TP or moves 2x TP distance)
                     should_activate_tsl = not trailing_sl_active and (
                          current_price >= base_tp  # or
                          # current_price >= entry_price * (1 + 1.5 * self.base_take_profit_pct) # Example: activate earlier
                     )
                     if should_activate_tsl:
                          new_trailing_sl = current_price * (1 - self.trailing_stop_loss_percentage)
                          # Ensure TSL starts at least at breakeven or slightly above
                          new_trailing_sl = max(new_trailing_sl, entry_price * 1.001)
                          position["trailing_stop_loss"] = new_trailing_sl
                          logger.info(
                              f"{Fore.MAGENTA}Trailing stop-loss ACTIVATED for LONG position (ID: {position_id}) at {new_trailing_sl:.2f}{Style.RESET_ALL}"
                          )
                          trailing_sl_active = True  # Mark as active for current loop logic
                          current_trailing_sl = new_trailing_sl  # Update for current loop logic

                     # Update existing TSL if price moves higher
                     elif trailing_sl_active:
                          potential_new_tsl = current_price * (1 - self.trailing_stop_loss_percentage)
                          if potential_new_tsl > current_trailing_sl:
                               position["trailing_stop_loss"] = potential_new_tsl
                               logger.info(
                                   f"{Fore.MAGENTA}Trailing stop-loss UPDATED for LONG position (ID: {position_id}) to {potential_new_tsl:.2f}{Style.RESET_ALL}"
                               )
                               # No need to update current_trailing_sl variable here as it's read next loop

            elif position_side == "sell":
                 # Stop Loss Check
                if current_price >= effective_stop_loss:
                    logger.info(
                        f"{Fore.RED}Stop-loss triggered for SHORT position (ID: {position_id}). "
                        f"Price ({current_price:.2f}) >= SL ({effective_stop_loss:.2f})"
                        f"{' (Trailing)' if trailing_sl_active else ''}{Style.RESET_ALL}"
                    )
                    exit_reason = "Stop Loss Hit"
                # Take Profit Check
                elif current_price <= base_tp:
                    logger.info(
                        f"{Fore.GREEN}Take-profit triggered for SHORT position (ID: {position_id}). "
                        f"Price ({current_price:.2f}) <= TP ({base_tp:.2f}){Style.RESET_ALL}"
                    )
                    exit_reason = "Take Profit Hit"
                 # Trailing Stop Activation/Update
                elif self.trailing_stop_loss_percentage > 0:
                      should_activate_tsl = not trailing_sl_active and (
                          current_price <= base_tp  # or
                          # current_price <= entry_price * (1 - 1.5 * self.base_take_profit_pct)
                      )
                      if should_activate_tsl:
                          new_trailing_sl = current_price * (1 + self.trailing_stop_loss_percentage)
                           # Ensure TSL starts at least at breakeven or slightly below
                          new_trailing_sl = min(new_trailing_sl, entry_price * 0.999)
                          position["trailing_stop_loss"] = new_trailing_sl
                          logger.info(
                              f"{Fore.MAGENTA}Trailing stop-loss ACTIVATED for SHORT position (ID: {position_id}) at {new_trailing_sl:.2f}{Style.RESET_ALL}"
                          )
                          trailing_sl_active = True
                          current_trailing_sl = new_trailing_sl

                      elif trailing_sl_active:
                          potential_new_tsl = current_price * (1 + self.trailing_stop_loss_percentage)
                          if potential_new_tsl < current_trailing_sl:
                               position["trailing_stop_loss"] = potential_new_tsl
                               logger.info(
                                   f"{Fore.MAGENTA}Trailing stop-loss UPDATED for SHORT position (ID: {position_id}) to {potential_new_tsl:.2f}{Style.RESET_ALL}"
                               )

            # --- 3. Execute Exit Order ---
            if exit_reason:
                logger.info(f"Placing closing order for {position_side.upper()} position (ID: {position_id}) due to: {exit_reason}")
                # Place a market order to close the position
                close_side = "sell" if position_side == "buy" else "buy"
                # Note: We use the original 'order_size' from the position dict here
                closing_order = self.place_order(
                    side=close_side,
                    order_size_quote=order_size * entry_price,  # Approx original quote value
                    confidence_level=confidence,  # Pass original confidence
                    order_type="market"  # Always use market order for reliable exit
                    # SL/TP params usually not needed for market close order, but check exchange specifics
                )
                if closing_order:
                    logger.info(f"{Fore.CYAN}Position (ID: {position_id}) closed successfully.{Style.RESET_ALL}")
                    # TODO: Implement PnL calculation here based on entry/exit prices/fees
                    # self.daily_pnl += calculated_pnl
                    self.open_positions.remove(position)
                else:
                    logger.error(f"{Fore.RED}Failed to place closing order for position (ID: {position_id}). Position remains open. Manual intervention may be required.{Style.RESET_ALL}")
                continue  # Move to the next position

    @retry_api_call()
    def cancel_all_open_orders(self) -> None:
        """Cancels all open limit orders for the trading symbol."""
        logger.info(f"Checking for open orders for {self.symbol} to cancel...")
        try:
            # Fetch only open orders
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            if not open_orders:
                logger.info(f"No open orders found for {self.symbol}.")
                return

            logger.info(f"{Fore.YELLOW}Found {len(open_orders)} open order(s) for {self.symbol}. Attempting cancellation...{Style.RESET_ALL}")
            cancelled_count = 0
            failed_count = 0
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
                    logger.info(f"{Fore.YELLOW}Cancelled order: ID {order['id']}, Side: {order['side']}, Price: {order['price']}, Amount: {order['amount']}{Style.RESET_ALL}")
                    cancelled_count += 1
                    time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
                except ccxt.OrderNotFound:
                    logger.warning(f"{Fore.YELLOW}Order {order['id']} already closed or cancelled.{Style.RESET_ALL}")
                    failed_count += 1  # Count as failed for summary, though it's not really an error
                except ccxt.NetworkError as e:
                     logger.error(f"{Fore.RED}Network error cancelling order {order['id']}: {e}. Will retry if decorator active.{Style.RESET_ALL}")
                     failed_count += 1
                     # Let retry decorator handle this if needed
                except Exception as e:
                    logger.error(f"{Fore.RED}Failed to cancel order {order['id']}: {e}{Style.RESET_ALL}", exc_info=True)
                    failed_count += 1

            logger.info(f"Order cancellation attempt finished. Cancelled: {cancelled_count}, Failed/Not Found: {failed_count}.")

        except Exception as e:
            logger.error(f"{Fore.RED}An error occurred during the order cancellation process: {e}{Style.RESET_ALL}", exc_info=True)

    # --- Main Trading Loop ---

    def run(self) -> None:
        """Starts the main trading loop of the bot."""
        logger.info(f"{Fore.CYAN}--- Starting Scalping Bot Main Loop (Symbol: {self.symbol}) ---{Style.RESET_ALL}")
        try:
            while True:
                self.iteration += 1
                logger.info(f"\n{Fore.BLUE}===== Iteration {self.iteration} ====={Style.RESET_ALL}")

                # --- 1. Fetch Data ---
                current_price = self.fetch_market_price()
                orderbook_imbalance = self.fetch_order_book()
                historical_data = self.fetch_historical_data()  # Fetches DataFrame

                # Basic data integrity check
                if current_price is None or historical_data is None or historical_data.empty:
                    logger.warning(
                        f"{Fore.YELLOW}Incomplete market data fetched (Price: {current_price}, "
                        f"HistData: {'OK' if historical_data is not None and not historical_data.empty else 'FAIL'}). "
                        f"Waiting {DEFAULT_SLEEP_INTERVAL_SECONDS}s...{Style.RESET_ALL}"
                    )
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue  # Skip iteration if core data is missing

                # --- 2. Calculate Indicators ---
                close_prices = historical_data['close']  # Use close prices Series
                indicators = {}
                indicators['ema'] = self.calculate_ema(close_prices, self.ema_period)
                indicators['rsi'] = self.calculate_rsi(close_prices, self.rsi_period)
                indicators['macd_line'], indicators['macd_signal'], indicators['macd_hist'] = self.calculate_macd(
                    close_prices, self.macd_short_period, self.macd_long_period, self.macd_signal_period
                )
                indicators['stoch_k'], indicators['stoch_d'] = self.calculate_stoch_rsi(
                    close_prices, self.rsi_period, self.stoch_rsi_period, self.stoch_rsi_k_period, self.stoch_rsi_d_period
                )
                # Volatility is primarily used for order sizing, but log it here too
                volatility = self.calculate_volatility(close_prices, self.volatility_window)

                # Log indicator state
                logger.info(f"Market State: Price={current_price:.2f}, Volatility={f'{volatility:.5f}' if volatility is not None else 'N/A'}")
                logger.info(f"Indicators: EMA={f'{indicators["ema"]:.2f}' if indicators['ema'] is not None else 'N/A'}, "
                            f"RSI={f'{indicators["rsi"]:.2f}' if indicators['rsi'] is not None else 'N/A'}, "
                            f"MACD={f'{indicators["macd_line"]:.4f}' if indicators['macd_line'] is not None else 'N/A'}, "
                            f"StochK={f'{indicators["stoch_k"]:.2f}' if indicators['stoch_k'] is not None else 'N/A'}")
                logger.info(f"Order Book Imbalance: {f'{orderbook_imbalance:.2f}' if orderbook_imbalance is not None else 'N/A'} "
                            f"(Threshold: {self.imbalance_threshold})")

                # --- 3. Calculate Order Size ---
                order_size_quote = self.calculate_order_size()
                if order_size_quote <= 0:
                    logger.warning(
                        f"{Fore.YELLOW}Order size calculated as 0. Cannot place trades. "
                        f"Waiting {DEFAULT_SLEEP_INTERVAL_SECONDS}s...{Style.RESET_ALL}"
                    )
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue

                # --- 4. Compute Trade Signal ---
                signal_score, reasons = self.compute_trade_signal_score(
                    current_price, indicators, orderbook_imbalance
                )
                logger.info(f"Trade Signal Score: {signal_score}")
                for reason in reasons:
                    logger.info(f" -> {reason}")

                # --- 5. Position Entry Logic ---
                can_open_new_position = len(self.open_positions) < self.max_open_positions

                if not can_open_new_position:
                     logger.info(
                        f"{Fore.YELLOW}Max open positions ({self.max_open_positions}) reached. "
                        f"Monitoring existing positions.{Style.RESET_ALL}"
                    )
                else:
                    # Check for LONG entry signal
                    if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS:
                        logger.info(f"{Fore.GREEN}Potential LONG entry signal detected (Score: {signal_score}){Style.RESET_ALL}")

                        # Adjust TP/SL based on confidence (signal score magnitude)
                        adjustment_factor = self.strong_signal_adjustment_factor if abs(signal_score) >= STRONG_SIGNAL_THRESHOLD_ABS else self.weak_signal_adjustment_factor
                        take_profit_pct = self.base_take_profit_pct * adjustment_factor
                        stop_loss_pct = self.base_stop_loss_pct * adjustment_factor

                        stop_loss_price = current_price * (1 - stop_loss_pct)
                        take_profit_price = current_price * (1 + take_profit_pct)
                        limit_entry_price = current_price * (1 - self.limit_order_entry_offset_pct_buy) if self.entry_order_type == "limit" else None

                        entry_order = self.place_order(
                            side="buy",
                            order_size_quote=order_size_quote,
                            confidence_level=signal_score,
                            order_type=self.entry_order_type,
                            limit_price=limit_entry_price,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )

                        if entry_order:
                            # Successfully placed order (or simulated)
                            actual_entry_price = entry_order.get('average') or entry_order.get('price')  # Use average if available (filled), else order price
                            self.open_positions.append({
                                "id": entry_order['id'],  # Store order ID
                                "side": "buy",
                                "size": entry_order['amount'],  # Store actual amount from order
                                "entry_price": actual_entry_price,
                                "entry_time": time.time(),
                                "stop_loss": stop_loss_price,  # Initial SL
                                "take_profit": take_profit_price,  # Initial TP
                                "confidence": signal_score,
                                # Trailing SL will be added later by manage_positions if activated
                            })
                            logger.info(f"{Fore.GREEN}---> Entered LONG position. Size: {entry_order['amount']}, Entry: {actual_entry_price:.2f}, "
                                        f"SL: {stop_loss_price:.2f} ({stop_loss_pct * 100:.2f}%), TP: {take_profit_price:.2f} ({take_profit_pct * 100:.2f}%){Style.RESET_ALL}")

                    # Check for SHORT entry signal
                    elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS:
                        logger.info(f"{Fore.RED}Potential SHORT entry signal detected (Score: {signal_score}){Style.RESET_ALL}")

                        adjustment_factor = self.strong_signal_adjustment_factor if abs(signal_score) >= STRONG_SIGNAL_THRESHOLD_ABS else self.weak_signal_adjustment_factor
                        take_profit_pct = self.base_take_profit_pct * adjustment_factor
                        stop_loss_pct = self.base_stop_loss_pct * adjustment_factor

                        stop_loss_price = current_price * (1 + stop_loss_pct)
                        take_profit_price = current_price * (1 - take_profit_pct)
                        limit_entry_price = current_price * (1 + self.limit_order_entry_offset_pct_sell) if self.entry_order_type == "limit" else None

                        entry_order = self.place_order(
                            side="sell",
                             order_size_quote=order_size_quote,
                            confidence_level=signal_score,
                            order_type=self.entry_order_type,
                            limit_price=limit_entry_price,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )

                        if entry_order:
                            actual_entry_price = entry_order.get('average') or entry_order.get('price')
                            self.open_positions.append({
                                "id": entry_order['id'],
                                "side": "sell",
                                "size": entry_order['amount'],
                                "entry_price": actual_entry_price,
                                "entry_time": time.time(),
                                "stop_loss": stop_loss_price,
                                "take_profit": take_profit_price,
                                "confidence": signal_score,
                            })
                            logger.info(f"{Fore.RED}---> Entered SHORT position. Size: {entry_order['amount']}, Entry: {actual_entry_price:.2f}, "
                                        f"SL: {stop_loss_price:.2f} ({stop_loss_pct * 100:.2f}%), TP: {take_profit_price:.2f} ({take_profit_pct * 100:.2f}%){Style.RESET_ALL}")

                    else:
                        logger.info(f"Neutral signal score ({signal_score}). No trade entry signal.")

                # --- 6. Manage Existing Positions ---
                self.manage_positions()

                # --- 7. Periodic Cleanup (Optional) ---
                # Cancel lingering limit orders occasionally to prevent clutter
                if self.iteration % 60 == 0:  # Example: every 60 iterations (~10 minutes)
                    self.cancel_all_open_orders()

                # --- 8. Wait for Next Iteration ---
                logger.debug(f"Iteration {self.iteration} complete. Waiting {DEFAULT_SLEEP_INTERVAL_SECONDS} seconds...")
                time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.warning(f"{Fore.YELLOW}\nCtrl+C detected. Shutting down bot gracefully...{Style.RESET_ALL}")
            # Cancel any remaining open orders before exiting
            self.cancel_all_open_orders()
            # Optionally close open positions here with market orders if desired
            # for position in list(self.open_positions): ... place market close ...
            logger.info("Scalping Bot has been stopped.")
        except Exception as e:
            logger.critical(f"{Fore.RED}A critical error occurred in the main loop: {e}{Style.RESET_ALL}", exc_info=True)
            # Attempt cleanup before exiting
            self.cancel_all_open_orders()
            logger.critical("Bot terminated due to critical error.")
        finally:
            # Ensure logs are written
            logging.shutdown()


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        bot = ScalpingBot(config_file=CONFIG_FILE_NAME)
        bot.run()
    except SystemExit:
        # Catch SystemExit exceptions (e.g., from config validation) to prevent traceback
        logger.info("Bot exited.")
    except Exception as e:
        # Catch any unexpected error during initialization
        logger.critical(f"{Fore.RED}Failed to initialize or run the bot: {e}{Style.RESET_ALL}", exc_info=True)
        logging.shutdown()
        sys.exit(1)  # Exit with error code
