


2025-05-01 04:01:41,044 - ScalpingBot - INFO - Loading configuration from config.yaml...
2025-05-01 04:01:41,057 - ScalpingBot - INFO - Configuration file loaded successfully.                                                        2025-05-01 04:01:41,058 - ScalpingBot - INFO - Validating configuration...                                                                    2025-05-01 04:01:41,058 - ScalpingBot - ERROR - Initialization failed due to invalid configuration: Missing required parameter 'exchange_id' in section 'exchange'.
%import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init

# --- Constants ---
CONFIG_FILE_DEFAULT = "config.yaml"
LOG_FILE = "scalping_bot.log"
API_MAX_RETRIES = 3
API_INITIAL_DELAY = 1.0  # seconds
MIN_HISTORICAL_DATA_BUFFER = 10 # Minimum extra data points beyond strict requirement

# Order Types
MARKET_ORDER = "market"
LIMIT_ORDER = "limit"

# Order Sides
BUY = "buy"
SELL = "sell"

# Configuration Keys (Example - can be expanded)
EXCHANGE_SECTION = "exchange"
TRADING_SECTION = "trading"
ORDER_BOOK_SECTION = "order_book"
INDICATORS_SECTION = "indicators"
RISK_MANAGEMENT_SECTION = "risk_management"

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Setup Logger ---
# Use a function for cleaner setup and potential reuse
def setup_logger(level: int = logging.DEBUG) -> logging.Logger:
    """Configures and returns the main application logger."""
    logger = logging.getLogger("ScalpingBot")
    logger.setLevel(level)

    # Prevent adding multiple handlers if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        logger.error(f"Failed to create log file handler for {LOG_FILE}: {e}")

    return logger

logger = setup_logger()


# --- API Retry Decorator ---
def retry_api_call(max_retries: int = API_MAX_RETRIES, initial_delay: float = API_INITIAL_DELAY):
    """Decorator to retry CCXT API calls with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(
                        f"Rate limit exceeded for {func.__name__}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                except ccxt.NetworkError as e:
                    logger.error(
                        f"Network error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                except ccxt.ExchangeError as e:
                    # Specific handling for non-critical exchange errors
                    if "Order does not exist" in str(e) or "Order not found" in str(e):
                        logger.warning(f"Order not found/does not exist for {func.__name__} (likely filled/cancelled): {e}. Returning None.")
                        return None
                    elif "Insufficient balance" in str(e) or "insufficient margin" in str(e).lower():
                         logger.error(f"Insufficient funds for {func.__name__}: {e}. Aborting call.")
                         # Re-raise specific exception if needed upstream, otherwise return None
                         # raise e # Option to propagate
                         return None # Current behavior matches original
                    else:
                        logger.error(
                            f"Exchange error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                            f"(Attempt {retries + 1}/{max_retries + 1})"
                        )
                except Exception as e: # Catch any other unexpected exception
                    logger.exception( # Use exception() to include traceback
                        f"Unexpected error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )

                # Only sleep and increment if we are going to retry
                if retries < max_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                else:
                    break # Exit loop if max retries reached

            logger.error(f"Max retries ({max_retries}) reached for API call {func.__name__}. Aborting call.")
            return None # Indicate failure after all retries

        return wrapper
    return decorator


# --- Configuration Validation Schema ---
# Define expected structure, types, and constraints for config.yaml
# This makes validation more declarative and easier to maintain.
CONFIG_SCHEMA = {
    EXCHANGE_SECTION: {
        "required": True,
        "params": {
            "exchange_id": {"type": str, "required": True, "non_empty": True},
            "api_key_env": {"type": str, "required": False, "default": "BYBIT_API_KEY"},
            "api_secret_env": {"type": str, "required": False, "default": "BYBIT_API_SECRET"},
        },
    },
    TRADING_SECTION: {
        "required": True,
        "params": {
            "symbol": {"type": str, "required": True, "non_empty": True},
            "simulation_mode": {"type": bool, "required": True},
            "entry_order_type": {"type": str, "required": True, "allowed": [MARKET_ORDER, LIMIT_ORDER]},
            "limit_order_offset_buy": {"type": (int, float), "required": True, "range": (0, float('inf'))},
            "limit_order_offset_sell": {"type": (int, float), "required": True, "range": (0, float('inf'))},
            "trade_loop_delay_seconds": {"type": (int, float), "required": False, "default": 10.0, "range": (0.1, float('inf'))},
        },
    },
    ORDER_BOOK_SECTION: {
        "required": True,
        "params": {
            "depth": {"type": int, "required": True, "range": (1, 1000)}, # Added practical upper limit
            "imbalance_threshold": {"type": (int, float), "required": True, "range": (0, float('inf'))},
        },
    },
    INDICATORS_SECTION: {
        "required": True,
        "params": {
            "timeframe": {"type": str, "required": False, "default": "1m"}, # Added timeframe config
            "volatility_window": {"type": int, "required": True, "range": (1, float('inf'))},
            "volatility_multiplier": {"type": (int, float), "required": True, "range": (0, float('inf'))},
            "ema_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "rsi_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "macd_short_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "macd_long_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "macd_signal_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "stoch_rsi_period": {"type": int, "required": True, "range": (1, float('inf'))},
        },
        "custom_validations": [
            lambda cfg: cfg["macd_short_period"] < cfg["macd_long_period"] or "MACD short period must be less than long period"
        ]
    },
    RISK_MANAGEMENT_SECTION: {
        "required": True,
        "params": {
            "order_size_percentage": {"type": (int, float), "required": True, "range": (0.0, 1.0)}, # Range 0-1
            "stop_loss_percentage": {"type": (int, float), "required": True, "range": (0.0, 1.0)}, # Range 0-1
            "take_profit_percentage": {"type": (int, float), "required": True, "range": (0.0, 1.0)}, # Range 0-1
            "max_open_positions": {"type": int, "required": True, "range": (1, float('inf'))},
            "time_based_exit_minutes": {"type": int, "required": True, "range": (1, float('inf'))},
            "trailing_stop_loss_percentage": {"type": (int, float), "required": True, "range": (0.0, 1.0)}, # Range 0-1
        },
    },
    "logging_level": { # Optional top-level param
        "required": False,
        "type": str,
        "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "default": "DEBUG"
    }
}


class ScalpingBot:
    """
    A scalping bot using CCXT for trading on cryptocurrency exchanges,
    based on technical indicators and order book analysis.
    """
    def __init__(self, config_file: Union[str, Path] = CONFIG_FILE_DEFAULT):
        """
        Initializes the ScalpingBot.

        Args:
            config_file: Path to the configuration file (YAML).

        Raises:
            FileNotFoundError: If the config file is not found.
            yaml.YAMLError: If the config file is invalid YAML.
            ValueError: If the configuration is invalid or incomplete.
            AttributeError: If the specified exchange is not supported by CCXT.
            ccxt.AuthenticationError: If API keys are invalid.
            ccxt.ExchangeError: If there's an issue connecting to the exchange.
            SystemExit: If initialization fails critically.
        """
        self.config_path = Path(config_file)
        self.config = self._load_and_validate_config()

        # Apply logging level from config if specified
        log_level_name = self.config.get("logging_level", "DEBUG").upper()
        log_level = getattr(logging, log_level_name, logging.DEBUG)
        setup_logger(level=log_level) # Reconfigure logger with potentially new level
        logger.info(f"Logger level set to {log_level_name}")

        # --- Assign validated config values to attributes ---
        # Using .get() with defaults where applicable, though validation should ensure presence
        # Exchange Config
        exch_cfg = self.config[EXCHANGE_SECTION]
        self.exchange_id: str = exch_cfg["exchange_id"]
        self.api_key_env: str = exch_cfg["api_key_env"]
        self.api_secret_env: str = exch_cfg["api_secret_env"]

        # Trading Config
        trade_cfg = self.config[TRADING_SECTION]
        self.symbol: str = trade_cfg["symbol"]
        self.simulation_mode: bool = trade_cfg["simulation_mode"]
        self.entry_order_type: str = trade_cfg["entry_order_type"]
        self.limit_order_offset_buy: float = float(trade_cfg["limit_order_offset_buy"])
        self.limit_order_offset_sell: float = float(trade_cfg["limit_order_offset_sell"])
        self.trade_loop_delay: float = float(trade_cfg["trade_loop_delay_seconds"])

        # Order Book Config
        ob_cfg = self.config[ORDER_BOOK_SECTION]
        self.order_book_depth: int = ob_cfg["depth"]
        self.imbalance_threshold: float = float(ob_cfg["imbalance_threshold"])

        # Indicators Config
        ind_cfg = self.config[INDICATORS_SECTION]
        self.timeframe: str = ind_cfg["timeframe"]
        self.volatility_window: int = ind_cfg["volatility_window"]
        self.volatility_multiplier: float = float(ind_cfg["volatility_multiplier"])
        self.ema_period: int = ind_cfg["ema_period"]
        self.rsi_period: int = ind_cfg["rsi_period"]
        self.macd_short_period: int = ind_cfg["macd_short_period"]
        self.macd_long_period: int = ind_cfg["macd_long_period"]
        self.macd_signal_period: int = ind_cfg["macd_signal_period"]
        self.stoch_rsi_period: int = ind_cfg["stoch_rsi_period"]

        # Risk Management Config
        risk_cfg = self.config[RISK_MANAGEMENT_SECTION]
        self.order_size_percentage: float = float(risk_cfg["order_size_percentage"])
        self.stop_loss_pct: float = float(risk_cfg["stop_loss_percentage"])
        self.take_profit_pct: float = float(risk_cfg["take_profit_percentage"])
        self.max_open_positions: int = risk_cfg["max_open_positions"]
        self.time_based_exit_minutes: int = risk_cfg["time_based_exit_minutes"]
        self.trailing_stop_loss_percentage: float = float(risk_cfg["trailing_stop_loss_percentage"])
        # --- End Config Assignment ---

        self.exchange: ccxt.Exchange = self._initialize_exchange() # Raises exceptions on failure
        self.market: Dict[str, Any] = self.exchange.market(self.symbol) # Get market details

        # --- State Variables ---
        self.open_positions: List[Dict[str, Any]] = [] # Store active positions/orders
        self.iteration: int = 0
        self.daily_pnl: float = 0.0 # Example metric, needs implementation
        self.last_candle_ts: Optional[int] = None # Track last processed candle timestamp

        logger.info(f"ScalpingBot initialized for {self.symbol} on {self.exchange_id}. Simulation Mode: {self.simulation_mode}")


    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Loads and validates the configuration file using CONFIG_SCHEMA."""
        logger.info(f"Loading configuration from {self.config_path}...")
        if not self.config_path.is_file():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Config file is not a valid YAML dictionary.")
            logger.info("Configuration file loaded successfully.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file: {e}")
            raise # Re-raise the original error

        logger.info("Validating configuration...")
        validated_config = {}
        for key, schema in CONFIG_SCHEMA.items():
            is_section = isinstance(schema, dict) and "params" in schema

            if is_section:
                section_name = key
                section_schema = schema
                if section_name not in config:
                    if section_schema.get("required", False):
                        raise ValueError(f"Missing required section '{section_name}' in config file.")
                    else:
                        logger.debug(f"Optional section '{section_name}' not found, skipping.")
                        validated_config[section_name] = {} # Add empty dict if optional section is missing
                        continue # Skip validation if section is missing but not required

                if not isinstance(config[section_name], dict):
                     raise ValueError(f"Section '{section_name}' must be a dictionary.")

                validated_section = {}
                section_config = config[section_name]

                for param, param_schema in section_schema["params"].items():
                    if param not in section_config:
                        if param_schema.get("required", False):
                            raise ValueError(f"Missing required parameter '{param}' in section '{section_name}'.")
                        elif "default" in param_schema:
                             validated_section[param] = param_schema["default"]
                             logger.debug(f"Using default value for '{param}' in '{section_name}': {param_schema['default']}")
                        else:
                             validated_section[param] = None # Parameter is optional without default
                             logger.debug(f"Optional parameter '{param}' not found in '{section_name}'.")
                             continue # Skip further checks if optional and not present

                    else:
                        value = section_config[param]
                        # Type check
                        expected_type = param_schema["type"]
                        if not isinstance(value, expected_type):
                            raise ValueError(f"Parameter '{param}' in '{section_name}' must be type {expected_type}, got {type(value)}.")

                        # Non-empty check for strings
                        if param_schema.get("non_empty", False) and isinstance(value, str) and not value.strip():
                             raise ValueError(f"Parameter '{param}' in '{section_name}' cannot be empty.")

                        # Allowed values check
                        allowed = param_schema.get("allowed")
                        if allowed and value not in allowed:
                            raise ValueError(f"Parameter '{param}' in '{section_name}' must be one of {allowed}, got '{value}'.")

                        # Range check
                        value_range = param_schema.get("range")
                        if value_range and not (value_range[0] <= value <= value_range[1]):
                             raise ValueError(f"Parameter '{param}' in '{section_name}' must be between {value_range[0]} and {value_range[1]}, got {value}.")

                        validated_section[param] = value # Assign validated value

                # Custom validations for the section
                custom_validations = section_schema.get("custom_validations", [])
                for validation_func in custom_validations:
                    result = validation_func(validated_section)
                    if isinstance(result, str): # Expect error message string on failure
                        raise ValueError(f"Custom validation failed for section '{section_name}': {result}")

                validated_config[section_name] = validated_section

            else: # Top-level parameter
                param_schema = schema
                if key not in config:
                    if param_schema.get("required", False):
                        raise ValueError(f"Missing required top-level parameter '{key}'.")
                    elif "default" in param_schema:
                        validated_config[key] = param_schema["default"]
                        logger.debug(f"Using default value for top-level parameter '{key}': {param_schema['default']}")
                    else:
                        validated_config[key] = None
                        logger.debug(f"Optional top-level parameter '{key}' not found.")
                else:
                    value = config[key]
                    expected_type = param_schema["type"]
                    if not isinstance(value, expected_type):
                        raise ValueError(f"Top-level parameter '{key}' must be type {expected_type}, got {type(value)}.")
                    allowed = param_schema.get("allowed")
                    if allowed and value not in allowed:
                        raise ValueError(f"Top-level parameter '{key}' must be one of {allowed}, got '{value}'.")
                    validated_config[key] = value

        logger.info("Configuration validated successfully.")
        return validated_config


    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and returns the CCXT exchange instance."""
        logger.info(f"Initializing exchange: {self.exchange_id.upper()}...")
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            api_key = os.getenv(self.api_key_env)
            api_secret = os.getenv(self.api_secret_env)

            if not self.simulation_mode and (not api_key or not api_secret):
                 logger.warning(
                     f"API Key ({self.api_key_env}) or Secret ({self.api_secret_env}) environment variables not set. "
                     "Proceeding without authentication (public endpoints only)."
                 )
                 # Allow proceeding for public data access, but trading will fail later.

            exchange_config = {
                "enableRateLimit": True,
                "options": {"defaultType": "linear"}, # Example option, specific to Bybit/some exchanges
                "recvWindow": 10000, # Example option
            }
            # Only add credentials if they exist and not in simulation mode
            if api_key and api_secret and not self.simulation_mode:
                 exchange_config["apiKey"] = api_key
                 exchange_config["secret"] = api_secret
                 logger.info("API credentials loaded from environment variables.")
            elif not self.simulation_mode:
                 logger.warning("Running in live mode without API credentials.")
            else:
                 logger.info("Running in simulation mode. No API credentials needed.")


            exchange = exchange_class(exchange_config)

            logger.debug("Loading markets...")
            exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id.upper()} successfully.")

            # Validate symbol exists on the exchange
            if self.symbol not in exchange.markets:
                available_symbols = list(exchange.markets.keys())
                logger.error(f"Symbol '{self.symbol}' not found on {self.exchange_id}.")
                logger.info(f"Available symbols sample: {available_symbols[:20]}...") # Log a sample
                raise ValueError(f"Symbol '{self.symbol}' not available on exchange '{self.exchange_id}'")

            # Check required capabilities (example)
            if not exchange.has['fetchOHLCV']:
                logger.warning(f"Exchange {self.exchange_id} does not support fetchOHLCV.")
            if not exchange.has['fetchOrderBook']:
                 logger.warning(f"Exchange {self.exchange_id} does not support fetchOrderBook.")
            # Add checks for createOrder, fetchBalance etc. if needed

            return exchange

        except AttributeError:
            logger.error(f"Exchange ID '{self.exchange_id}' is not a valid CCXT exchange.")
            raise # Re-raise
        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication Error with {self.exchange_id}: {e}. Check API keys ({self.api_key_env}, {self.api_secret_env}).")
            raise # Re-raise
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange Not Available: {self.exchange_id} might be down or unreachable. {e}")
            raise # Re-raise
        except ccxt.NetworkError as e:
             logger.error(f"Network error during exchange initialization: {e}")
             raise # Re-raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during exchange initialization: {e}")
            raise # Re-raise unexpected errors


    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the last traded price for the symbol."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        # Prefer 'last', fallback to 'close', then 'bid' as last resort
        price = ticker.get("last") or ticker.get("close") or ticker.get("bid")
        if price is not None:
            logger.debug(f"Fetched market price: {price}")
            return float(price)
        else:
            logger.warning(f"Market price ('last', 'close', 'bid') unavailable in ticker for {self.symbol}. Ticker: {ticker}")
            return None

    @retry_api_call()
    def fetch_order_book(self) -> Optional[float]:
        """Fetches order book and calculates the bid/ask volume imbalance ratio."""
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            logger.warning(f"Order book data missing bids or asks for {self.symbol}. Bids: {len(bids)}, Asks: {len(asks)}")
            return None

        # Calculate total volume within the specified depth
        bid_volume = sum(bid[1] for bid in bids)
        ask_volume = sum(ask[1] for ask in asks)

        if ask_volume <= 0:
            # Handle division by zero: Infinite imbalance if bids exist, neutral (1.0) if neither exist (though checked above)
            imbalance_ratio = float("inf") if bid_volume > 0 else 1.0
        else:
            imbalance_ratio = bid_volume / ask_volume

        logger.debug(f"Order Book - Top {self.order_book_depth} levels: "
                     f"Bid Vol: {bid_volume:.4f}, Ask Vol: {ask_volume:.4f}, "
                     f"Bid/Ask Ratio: {imbalance_ratio:.2f}")
        return imbalance_ratio


    @retry_api_call()
    def fetch_historical_data(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV data."""
        # Determine the minimum number of candles required for all indicators
        min_required = max(
            self.volatility_window,
            self.ema_period,
            self.rsi_period + 1, # RSI needs period + 1 for diff
            self.macd_long_period + self.macd_signal_period, # MACD needs long + signal for full calculation
            self.stoch_rsi_period + 6, # Stoch RSI needs period + 3 (k) + 3 (d)
            MIN_HISTORICAL_DATA_BUFFER # Ensure a baseline minimum
        )

        fetch_limit = limit if limit is not None else min_required + MIN_HISTORICAL_DATA_BUFFER # Fetch slightly more than needed

        logger.debug(f"Fetching {fetch_limit} historical {self.timeframe} candles for {self.symbol} (min required: {min_required})...")

        if not self.exchange.has['fetchOHLCV']:
             logger.error(f"Exchange {self.exchange_id} does not support fetchOHLCV. Cannot fetch historical data.")
             return None

        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=fetch_limit)

        if not ohlcv:
            logger.warning(f"Historical OHLCV data unavailable for {self.symbol} with timeframe {self.timeframe}.")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Drop potential duplicates just in case API returns overlapping data
        df = df[~df.index.duplicated(keep='first')]

        if len(df) < min_required:
            logger.error(f"Insufficient historical data fetched. Requested {fetch_limit}, got {len(df)}, minimum required {min_required}. Cannot proceed with calculations.")
            return None

        logger.debug(f"Historical data fetched successfully. Shape: {df.shape}, Time Range: {df.index.min()} to {df.index.max()}")
        return df

    # --- Indicator Calculations ---

    def _calculate_indicator(self, data: Optional[pd.Series], required_length: int, calc_func, name: str, **kwargs) -> Optional[Any]:
        """Helper to check data length before calculating an indicator."""
        if data is None or len(data) < required_length:
            logger.warning(f"Insufficient data for {name} calculation (need {required_length}, got {len(data) if data is not None else 0})")
            return None
        try:
            result = calc_func(data, **kwargs)
            if isinstance(result, pd.Series): # Get last value if series is returned
                 result_val = result.iloc[-1]
            else:
                 result_val = result

            if pd.isna(result_val):
                logger.warning(f"{name} calculation resulted in NaN.")
                return None
            # Log scalar results, handle tuples separately if needed
            if isinstance(result_val, (float, int)):
                 logger.debug(f"Calculated {name}: {result_val:.4f}") # Adjust precision as needed
            else:
                 logger.debug(f"Calculated {name}: {result_val}")
            return result_val
        except Exception as e:
            logger.error(f"Error calculating {name}: {e}")
            return None


    def calculate_volatility(self, data: pd.DataFrame) -> Optional[float]:
        """Calculates the rolling log return standard deviation (volatility)."""
        if data is None or 'close' not in data.columns: return None
        required = self.volatility_window + 1 # Need one previous point for shift
        return self._calculate_indicator(
            data['close'], required,
            lambda s: np.log(s / s.shift(1)).rolling(window=self.volatility_window).std(),
            f"Volatility (window {self.volatility_window})"
        )

    def calculate_ema(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """Calculates the Exponential Moving Average (EMA)."""
        if data is None or 'close' not in data.columns: return None
        return self._calculate_indicator(
            data['close'], period,
            lambda s: s.ewm(span=period, adjust=False).mean(),
            f"EMA (period {period})"
        )

    def calculate_rsi(self, data: pd.DataFrame) -> Optional[float]:
        """Calculates the Relative Strength Index (RSI)."""
        if data is None or 'close' not in data.columns: return None
        required = self.rsi_period + 1 # For diff()
        def _rsi_calc(series: pd.Series):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10) # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi

        return self._calculate_indicator(
            data['close'], required, _rsi_calc, f"RSI (period {self.rsi_period})"
        )

    def calculate_macd(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        if data is None or 'close' not in data.columns: return None, None, None
        required = self.macd_long_period + self.macd_signal_period # Rough estimate

        close_series = data['close']
        if len(close_series) < required:
             logger.warning(f"Insufficient data for MACD calculation (need ~{required}, got {len(close_series)})")
             return None, None, None

        try:
            ema_short = close_series.ewm(span=self.macd_short_period, adjust=False).mean()
            ema_long = close_series.ewm(span=self.macd_long_period, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=self.macd_signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]
            hist_val = histogram.iloc[-1]

            if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
                logger.warning("MACD calculation resulted in NaN values.")
                return None, None, None

            logger.debug(f"MACD: {macd_val:.4f}, Signal: {signal_val:.4f}, Histogram: {hist_val:.4f}")
            return macd_val, signal_val, hist_val
        except Exception as e:
             logger.error(f"Error calculating MACD: {e}")
             return None, None, None


    def calculate_stoch_rsi(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Calculates the Stochastic RSI."""
        if data is None or 'close' not in data.columns: return None, None
        # Stoch RSI needs RSI calculation + rolling window + k smoothing + d smoothing
        required = self.rsi_period + self.stoch_rsi_period + 3 + 3 # rsi_period for RSI, stoch_rsi_period for rolling, 3 for k, 3 for d

        close_series = data['close']
        if len(close_series) < required:
             logger.warning(f"Insufficient data for Stoch RSI calculation (need ~{required}, got {len(close_series)})")
             return None, None

        try:
            # Calculate RSI first
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi_series = 100 - (100 / (1 + rs))

            # Calculate Stoch RSI
            min_rsi = rsi_series.rolling(window=self.stoch_rsi_period).min()
            max_rsi = rsi_series.rolling(window=self.stoch_rsi_period).max()
            stoch_rsi_k_raw = 100 * (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-10) # Avoid div by zero

            # Smooth K and D
            k = stoch_rsi_k_raw.rolling(window=3).mean()
            d = k.rolling(window=3).mean()

            k_val = k.iloc[-1]
            d_val = d.iloc[-1]

            if pd.isna(k_val) or pd.isna(d_val):
                logger.warning("Stochastic RSI calculation resulted in NaN.")
                return None, None

            logger.debug(f"Calculated Stochastic RSI (period {self.stoch_rsi_period}) - K: {k_val:.2f}, D: {d_val:.2f}")
            return k_val, d_val
        except Exception as e:
             logger.error(f"Error calculating Stochastic RSI: {e}")
             return None, None

    # --- Balance and Order Sizing ---

    @retry_api_call()
    def fetch_balance(self) -> Optional[float]:
        """Fetches the free balance for the quote currency."""
        if self.simulation_mode:
            logger.debug("Simulation mode: Using simulated balance of 10000.")
            return 10000.0 # Return a fixed balance for simulation

        try:
            balance = self.exchange.fetch_balance()
            quote_currency = self.market.get("quote")
            if not quote_currency:
                 logger.error("Could not determine quote currency from market data.")
                 return None

            # Use 'free' balance for placing new orders
            free_balance = balance.get("free", {}).get(quote_currency)

            if free_balance is None:
                 logger.warning(f"Could not find free balance for quote currency '{quote_currency}' in balance response: {balance}")
                 # Fallback: try total balance if free is missing? Or return None?
                 total_balance = balance.get("total", {}).get(quote_currency, 0.0)
                 logger.warning(f"Attempting to use total balance: {total_balance}")
                 free_balance = total_balance # Be cautious with this fallback

            logger.debug(f"Fetched free balance for {quote_currency}: {free_balance}")
            return float(free_balance)

        except ccxt.AuthenticationError:
             logger.error("Authentication required to fetch balance. Check API keys.")
             return None
        except Exception as e:
            logger.error(f"Could not retrieve {self.market.get('quote', 'quote currency')} balance: {e}")
            return None

    def calculate_order_size(self, price: float, volatility: Optional[float]) -> Optional[float]:
        """
        Calculates the order size in the base currency, considering risk percentage,
        volatility adjustment, and exchange minimums.

        Args:
            price: The current market price.
            volatility: The calculated volatility (optional).

        Returns:
            The calculated order size in base currency, or None if calculation fails.
        """
        if price <= 0:
            logger.error(f"Invalid price ({price}) for order size calculation.")
            return None

        balance = self.fetch_balance()
        if balance is None or balance <= 0:
            logger.error(f"Could not calculate order size due to invalid or zero balance ({balance}).")
            return None

        # 1. Calculate base size based on risk percentage of balance
        base_size_quote = balance * self.order_size_percentage
        logger.debug(f"Base order size (quote): {base_size_quote:.4f} ({self.order_size_percentage*100:.2f}% of {balance:.2f})")

        # 2. Adjust size based on volatility (optional)
        if volatility is not None and volatility > 0 and self.volatility_multiplier > 0:
            # Reduce size more in higher volatility; adjustment factor < 1
            adjustment_factor = 1 / (1 + (volatility * self.volatility_multiplier))
            adjusted_size_quote = base_size_quote * adjustment_factor
            logger.debug(f"Volatility adjustment factor: {adjustment_factor:.4f} (Volatility: {volatility:.5f})")
        else:
            adjusted_size_quote = base_size_quote
            logger.debug("No volatility adjustment applied (Volatility N/A or multiplier is 0).")

        # 3. Apply a maximum capital cap per trade (e.g., 5% of balance - make configurable?)
        # max_capital_per_trade_pct = 0.05 # Example: Hardcoded cap - consider adding to config
        # max_size_quote = balance * max_capital_per_trade_pct
        # final_size_quote = min(adjusted_size_quote, max_size_quote)
        # logger.debug(f"Applied max capital cap ({max_capital_per_trade_pct*100}%). Size (quote): {final_size_quote:.4f}")

        # Using adjusted size directly for now, as per original logic's effective cap via order_size_percentage
        final_size_quote = adjusted_size_quote

        # 4. Convert quote size to base size
        final_size_base = final_size_quote / price
        logger.debug(f"Calculated base size before precision/min checks: {final_size_base:.8f} {self.market['base']}")

        # 5. Apply exchange precision and minimum order size constraints
        amount_precision = self.market.get("precision", {}).get("amount")
        min_amount = self.market.get("limits", {}).get("amount", {}).get("min")

        # Apply amount precision first
        if amount_precision is not None:
            try:
                precise_size_base_str = self.exchange.amount_to_precision(self.symbol, final_size_base)
                precise_size_base = float(precise_size_base_str)
                if precise_size_base <= 0:
                    logger.warning(f"Order size became zero after applying amount precision. Initial base size: {final_size_base:.8f}")
                    return None # Cannot place zero size order
                logger.debug(f"Applied amount precision ({amount_precision}): {precise_size_base:.8f}")
                final_size_base = precise_size_base
            except Exception as e:
                 logger.error(f"Failed to apply amount precision: {e}")
                 # Decide whether to proceed with un-precised value or fail
                 return None # Safer to fail

        # Check against minimum order size
        if min_amount is not None and final_size_base < min_amount:
            logger.warning(f"Calculated order size {final_size_base:.8f} is below exchange minimum {min_amount}. Checking if minimum is within risk.")
            min_size_quote = min_amount * price
            # Check if using the minimum size exceeds our initial risk capital
            if min_size_quote <= base_size_quote: # Compare against the initial risk budget
                logger.info(f"Adjusting order size to exchange minimum: {min_amount} {self.market['base']}")
                final_size_base = min_amount
                # Re-apply precision just in case min_amount itself needs it (unlikely but possible)
                if amount_precision is not None:
                    final_size_base = float(self.exchange.amount_to_precision(self.symbol, final_size_base))

            else:
                logger.warning(f"Minimum order size ({min_amount} {self.market['base']} = ~{min_size_quote:.2f} {self.market['quote']}) "
                               f"exceeds the allocated risk capital ({base_size_quote:.2f} {self.market['quote']}). Skipping trade.")
                return None

        # Final check for zero size
        if final_size_base <= 0:
             logger.error("Final calculated order size is zero or negative. Skipping trade.")
             return None

        logger.info(
            f"Final Calculated Order Size: {final_size_base:.8f} {self.market['base']} "
            f"(Quote Value: ~{final_size_base * price:.2f} {self.market['quote']})"
        )
        return final_size_base


    # --- Trade Execution Logic ---

    def compute_trade_signal_score(self, price: float, indicators: dict, orderbook_imbalance: Optional[float]) -> Tuple[int, List[str]]:
        """Computes a simple score based on indicator signals and order book imbalance."""
        score = 0
        reasons = []

        # --- Extract Indicator Values ---
        # Use .get() to handle potential None values gracefully
        ema = indicators.get("ema")
        rsi = indicators.get("rsi")
        macd_line = indicators.get("macd_line")
        macd_signal = indicators.get("macd_signal")
        # macd_hist = indicators.get("macd_hist") # Histogram not used in original logic
        stoch_rsi_k = indicators.get("stoch_rsi_k")
        stoch_rsi_d = indicators.get("stoch_rsi_d")

        # --- Order Book Imbalance ---
        if orderbook_imbalance is not None:
            if orderbook_imbalance > self.imbalance_threshold:
                score += 1
                reasons.append(f"OB Imbalance: Strong bid pressure (Ratio: {orderbook_imbalance:.2f} > {self.imbalance_threshold:.2f})")
            elif self.imbalance_threshold > 0 and orderbook_imbalance < (1 / self.imbalance_threshold): # Avoid division by zero if threshold is 0
                score -= 1
                reasons.append(f"OB Imbalance: Strong ask pressure (Ratio: {orderbook_imbalance:.2f} < {1/self.imbalance_threshold:.2f})")
            else:
                reasons.append(f"OB Imbalance: Neutral pressure (Ratio: {orderbook_imbalance:.2f})")
        else:
             reasons.append("OB Imbalance: N/A")

        # --- EMA Trend ---
        if ema is not None:
            if price > ema:
                score += 1
                reasons.append(f"Trend: Price > EMA ({price:.2f} > {ema:.2f}) (Bullish)")
            elif price < ema:
                score -= 1
                reasons.append(f"Trend: Price < EMA ({price:.2f} < {ema:.2f}) (Bearish)")
            else:
                reasons.append(f"Trend: Price == EMA ({price:.2f})")
        else:
             reasons.append("Trend: EMA N/A")

        # --- RSI Overbought/Oversold ---
        if rsi is not None:
            if rsi < 30:
                score += 1 # Potential buy signal (oversold)
                reasons.append(f"Momentum: RSI < 30 ({rsi:.2f}) (Oversold)")
            elif rsi > 70:
                score -= 1 # Potential sell signal (overbought)
                reasons.append(f"Momentum: RSI > 70 ({rsi:.2f}) (Overbought)")
            else:
                reasons.append(f"Momentum: RSI Neutral ({rsi:.2f})")
        else:
             reasons.append("Momentum: RSI N/A")

        # --- MACD Crossover ---
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                score += 1 # Bullish crossover
                reasons.append(f"Momentum: MACD > Signal ({macd_line:.4f} > {macd_signal:.4f}) (Bullish)")
            elif macd_line < macd_signal:
                score -= 1 # Bearish crossover
                reasons.append(f"Momentum: MACD < Signal ({macd_line:.4f} < {macd_signal:.4f}) (Bearish)")
            else:
                reasons.append(f"Momentum: MACD == Signal ({macd_line:.4f})")
        else:
             reasons.append("Momentum: MACD N/A")

        # --- Stochastic RSI Overbought/Oversold ---
        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            # Simple check: both lines in extreme zones
            if stoch_rsi_k < 20 and stoch_rsi_d < 20:
                score += 1 # Oversold
                reasons.append(f"Momentum: Stoch RSI < 20 (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (Oversold)")
            elif stoch_rsi_k > 80 and stoch_rsi_d > 80:
                score -= 1 # Overbought
                reasons.append(f"Momentum: Stoch RSI > 80 (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (Overbought)")
            # Could add crossover logic here too (e.g., K crossing D)
            else:
                reasons.append(f"Momentum: Stoch RSI Neutral (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f})")
        else:
             reasons.append("Momentum: Stoch RSI N/A")

        logger.debug(f"Signal Score: {score}, Reasons: {'; '.join(reasons)}")
        return score, reasons


    @retry_api_call()
    def place_order(
        self,
        side: str, # Use constants BUY or SELL
        order_size: float,
        order_type: str = MARKET_ORDER, # Use constants MARKET_ORDER or LIMIT_ORDER
        price: Optional[float] = None, # Required for limit orders
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Places a trade order (simulated or live)."""

        # --- Input Validation ---
        if side not in [BUY, SELL]:
            logger.error(f"Invalid order side: '{side}'. Must be '{BUY}' or '{SELL}'.")
            return None
        if order_type not in [MARKET_ORDER, LIMIT_ORDER]:
            logger.error(f"Invalid order type: '{order_type}'. Must be '{MARKET_ORDER}' or '{LIMIT_ORDER}'.")
            return None
        if order_type == LIMIT_ORDER and price is None:
            logger.error(f"Price is required for {LIMIT_ORDER} orders.")
            return None
        if order_size <= 0:
            logger.error(f"Order size must be positive, got {order_size}.")
            return None

        # --- Prepare Order Parameters ---
        try:
            order_size_str = self.exchange.amount_to_precision(self.symbol, order_size)
            price_str = self.exchange.price_to_precision(self.symbol, price) if price else None
            sl_price_str = self.exchange.price_to_precision(self.symbol, stop_loss_price) if stop_loss_price else None
            tp_price_str = self.exchange.price_to_precision(self.symbol, take_profit_price) if take_profit_price else None
        except Exception as e:
             logger.error(f"Error formatting order parameters for precision: {e}")
             return None

        # CCXT specific parameters for SL/TP (may vary by exchange)
        params = {}
        if self.exchange.id == 'bybit': # Example: Bybit uses specific params
            if stop_loss_price:
                params["stopLoss"] = sl_price_str
            if take_profit_price:
                params["takeProfit"] = tp_price_str
        # Add other exchange-specific param structures here if needed
        # else: # Generic attempt (might not work on all exchanges)
        #     if stop_loss_price: params["stopLossPrice"] = sl_price_str
        #     if take_profit_price: params["takeProfitPrice"] = tp_price_str

        # --- Simulation Mode ---
        if self.simulation_mode:
            simulated_fill_price = price if order_type == LIMIT_ORDER else self.fetch_market_price()
            if simulated_fill_price is None and order_type == MARKET_ORDER:
                logger.error("[SIMULATION] Could not fetch market price for simulated market order.")
                return None
            elif simulated_fill_price is None and order_type == LIMIT_ORDER:
                 # Should not happen due to earlier check, but as safeguard
                 logger.error("[SIMULATION] Price is None for simulated limit order.")
                 return None

            simulated_fill_price = float(simulated_fill_price) # Ensure float
            order_size_float = float(order_size_str)
            cost = order_size_float * simulated_fill_price

            # Simulate fees (example: 0.1%) - make configurable
            simulated_fee_rate = 0.001
            fee = cost * simulated_fee_rate
            simulated_cost_with_fee = cost + fee if side == BUY else cost - fee # Adjust cost based on fee

            trade_details = {
                "id": f"sim_{int(time.time() * 1000)}_{side}",
                "symbol": self.symbol,
                "status": "closed", # Simulate immediate fill
                "side": side,
                "type": order_type,
                "amount": order_size_float,
                "price": simulated_fill_price, # For limit orders, this is the requested price
                "average": simulated_fill_price, # Actual fill price
                "cost": simulated_cost_with_fee, # Cost including simulated fee
                "filled": order_size_float,
                "remaining": 0.0,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.Timestamp.now(tz='UTC').isoformat(),
                "fee": {"cost": fee, "currency": self.market['quote']},
                "info": { # Mimic structure of real order where possible
                    "stopLoss": sl_price_str,
                    "takeProfit": tp_price_str,
                    "orderType": order_type.capitalize(),
                    "execType": "Trade", # Simulate execution type
                    "is_simulated": True,
                },
            }
            log_price = price_str if order_type == LIMIT_ORDER else f"Market (~{simulated_fill_price:.{self.market['precision']['price']}f})"
            logger.info(
                f"{Fore.YELLOW}[SIMULATION] Order Placed: ID: {trade_details['id']}, Type: {trade_details['type']}, "
                f"Side: {trade_details['side'].upper()}, Size: {trade_details['amount']:.{self.market['precision']['amount']}f} {self.market['base']}, "
                f"Fill Price: {log_price}, SL: {sl_price_str or 'N/A'}, TP: {tp_price_str or 'N/A'}, "
                f"Simulated Cost: {trade_details['cost']:.2f} {self.market['quote']} (Fee: {fee:.4f})"
            )
            return trade_details

        # --- Live Trading Mode ---
        else:
            order = None
            order_size_float = float(order_size_str)
            try:
                if order_type == MARKET_ORDER:
                    logger.info(f"{Fore.CYAN}Placing LIVE Market {side.upper()} order for {order_size_str} {self.market['base']} with params: {params}")
                    order = self.exchange.create_market_order(self.symbol, side, order_size_float, params=params)
                elif order_type == LIMIT_ORDER:
                    price_float = float(price_str) # Ensure price is float
                    logger.info(f"{Fore.CYAN}Placing LIVE Limit {side.upper()} order for {order_size_str} {self.market['base']} at {price_str} with params: {params}")
                    order = self.exchange.create_limit_order(self.symbol, side, order_size_float, price_float, params=params)

                if order:
                    # Log details from the returned order structure
                    order_id = order.get('id', 'N/A')
                    order_status = order.get('status', 'N/A')
                    avg_price = order.get('average')
                    filled_amount = order.get('filled', 0.0)
                    cost = order.get('cost') # Cost might include fees depending on exchange

                    log_price_actual = avg_price or order.get('price') # Use average if available, else requested price
                    log_price_display = f"{log_price_actual:.{self.market['precision']['price']}f}" if log_price_actual else (price_str if order_type == LIMIT_ORDER else 'Market')

                    logger.info(
                        f"{Fore.GREEN}LIVE Order Placed Successfully: ID: {order_id}, Type: {order.get('type')}, Side: {order.get('side', '').upper()}, "
                        f"Amount: {order.get('amount'):.{self.market['precision']['amount']}f}, Filled: {filled_amount:.{self.market['precision']['amount']}f} {self.market['base']}, "
                        f"Avg Price: {log_price_display}, Cost: {cost:.2f} {self.market['quote'] if cost else ''}, "
                        f"SL: {params.get('stopLoss', 'N/A')}, TP: {params.get('takeProfit', 'N/A')}, Status: {order_status}"
                    )
                    # Add the placed order details (especially ID) to track open positions if needed
                    # self.open_positions.append(order) # Manage this in the main loop
                    return order
                else:
                    # This case might occur if the API call succeeded (no exception) but CCXT returned None/empty
                    logger.error("LIVE Order placement call returned no result or an empty order structure.")
                    return None

            except ccxt.InsufficientFunds as e:
                 logger.error(f"{Fore.RED}LIVE Order Failed: Insufficient funds. {e}")
                 return None # Return None, retry decorator already handled retries if applicable
            except ccxt.ExchangeError as e:
                 logger.error(f"{Fore.RED}LIVE Order Failed: Exchange error. {e}")
                 return None
            except ccxt.NetworkError as e:
                 logger.error(f"{Fore.RED}LIVE Order Failed: Network error. {e}")
                 return None # Should have been caught by retry, but as a safeguard
            except Exception as e:
                 logger.exception(f"{Fore.RED}LIVE Order Failed: An unexpected error occurred during order placement.")
                 return None


    # --- Main Loop Logic (Example Structure) ---

    def run(self):
        """Main execution loop for the bot."""
        logger.info(f"Starting Scalping Bot - {self.symbol} - Loop Delay: {self.trade_loop_delay}s")
        if self.simulation_mode:
             logger.warning("--- RUNNING IN SIMULATION MODE ---")

        while True:
            self.iteration += 1
            logger.info(f"\n----- Iteration {self.iteration} -----")
            start_time = time.time()

            try:
                # 1. Fetch Data
                current_price = self.fetch_market_price()
                orderbook_imbalance = self.fetch_order_book()
                historical_data = self.fetch_historical_data() # Fetches based on indicator needs

                if current_price is None or historical_data is None:
                    logger.warning("Could not fetch required market data. Skipping iteration.")
                    time.sleep(self.trade_loop_delay)
                    continue

                # Check if new candle has arrived (optional, prevents redundant calculations)
                current_candle_ts = historical_data.index[-1].value // 10**9 # Timestamp in seconds
                if self.last_candle_ts is not None and current_candle_ts <= self.last_candle_ts:
                     logger.debug(f"No new candle data since last iteration ({pd.Timestamp(self.last_candle_ts, unit='s', tz='UTC')}). Waiting...")
                     time.sleep(self.trade_loop_delay) # Wait before retrying
                     continue
                self.last_candle_ts = current_candle_ts


                # 2. Calculate Indicators
                volatility = self.calculate_volatility(historical_data)
                ema = self.calculate_ema(historical_data, self.ema_period)
                rsi = self.calculate_rsi(historical_data)
                macd_line, macd_signal, macd_hist = self.calculate_macd(historical_data)
                stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(historical_data)

                indicators = {
                    "volatility": volatility, "ema": ema, "rsi": rsi,
                    "macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
                    "stoch_rsi_k": stoch_rsi_k, "stoch_rsi_d": stoch_rsi_d,
                }

                # 3. Check Open Positions & Manage Exits (SL/TP/Trailing/Time)
                # TODO: Implement position management logic here
                # - Fetch open orders/positions from exchange (or track internally)
                # - Check if SL/TP hit based on current_price
                # - Check time-based exits
                # - Update trailing stops

                # 4. Generate Trade Signal
                if len(self.open_positions) >= self.max_open_positions:
                    logger.info(f"Max open positions ({self.max_open_positions}) reached. Holding.")
                else:
                    score, reasons = self.compute_trade_signal_score(current_price, indicators, orderbook_imbalance)

                    # Define entry thresholds (example: require a strong signal)
                    buy_threshold = 2
                    sell_threshold = -2

                    trade_side = None
                    if score >= buy_threshold:
                        trade_side = BUY
                        logger.info(f"{Fore.GREEN}Potential BUY signal (Score: {score}). Reasons: {'; '.join(reasons)}")
                    elif score <= sell_threshold:
                        trade_side = SELL
                        logger.info(f"{Fore.RED}Potential SELL signal (Score: {score}). Reasons: {'; '.join(reasons)}")
                    else:
                        logger.info(f"Neutral signal (Score: {score}). Holding.")

                    # 5. Place Order if Signal Strong Enough
                    if trade_side:
                        order_size = self.calculate_order_size(current_price, volatility)

                        if order_size is not None and order_size > 0:
                            # Calculate SL/TP prices
                            sl_price, tp_price = None, None
                            if trade_side == BUY:
                                if self.stop_loss_pct > 0:
                                    sl_price = current_price * (1 - self.stop_loss_pct)
                                if self.take_profit_pct > 0:
                                    tp_price = current_price * (1 + self.take_profit_pct)
                                # Adjust limit order price slightly below current for buys
                                entry_price = current_price * (1 - self.limit_order_offset_buy) if self.entry_order_type == LIMIT_ORDER else None
                            else: # SELL
                                if self.stop_loss_pct > 0:
                                    sl_price = current_price * (1 + self.stop_loss_pct)
                                if self.take_profit_pct > 0:
                                    tp_price = current_price * (1 - self.take_profit_pct)
                                # Adjust limit order price slightly above current for sells
                                entry_price = current_price * (1 + self.limit_order_offset_sell) if self.entry_order_type == LIMIT_ORDER else None

                            # Place the order
                            order_result = self.place_order(
                                side=trade_side,
                                order_size=order_size,
                                order_type=self.entry_order_type,
                                price=entry_price, # None for market orders
                                stop_loss_price=sl_price,
                                take_profit_price=tp_price,
                            )

                            if order_result:
                                logger.info(f"Successfully placed {trade_side.upper()} order.")
                                # TODO: Add order_result to self.open_positions for tracking
                                # Need a robust way to track simulated vs real positions/orders
                                # self.open_positions.append(order_result)
                            else:
                                logger.error(f"Failed to place {trade_side.upper()} order.")
                        else:
                             logger.warning("Order size calculation failed or resulted in zero. No order placed.")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                break
            except ccxt.AuthenticationError as e:
                 logger.critical(f"CRITICAL: Authentication failed during main loop: {e}. Check API keys. Exiting.")
                 break # Exit loop on critical auth error
            except Exception as e:
                logger.exception("An unexpected error occurred in the main loop.")
                # Decide whether to continue or break on general errors
                # break # Option to stop on any error

            # --- Loop Delay ---
            end_time = time.time()
            elapsed = end_time - start_time
            sleep_time = max(0, self.trade_loop_delay - elapsed)
            logger.debug(f"Iteration took {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)

        logger.info("Scalping Bot stopped.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        bot = ScalpingBot(config_file=CONFIG_FILE_DEFAULT)
        bot.run()
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    except (ValueError, yaml.YAMLError) as e:
        logger.error(f"Initialization failed due to invalid configuration: {e}")
        sys.exit(1)
    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError) as e:
         logger.error(f"Initialization failed due to exchange connection issue: {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected critical error occurred during initialization or runtime: {e}")
        sys.exit(1)
