import logging
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
API_RETRY_DELAY = 1.0  # seconds
MIN_HISTORICAL_DATA_BUFFER = 10
MAX_ORDER_BOOK_DEPTH = 1000  # Practical limit

# Order Types and Sides (Enums for better type hinting and readability)
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

# Configuration Sections
CONFIG_SECTIONS = {
    "exchange": {"required": True},
    "trading": {"required": True},
    "order_book": {"required": True},
    "indicators": {"required": True},
    "risk_management": {"required": True},
    "logging_level": {"required": False, "default": "INFO"}, # Moved logging level to config
}

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Setup Logger ---
def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures and returns the main application logger."""
    logger = logging.getLogger("ScalpingBot")
    log_level = log_level.upper()
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(LOG_FILE, mode="a") # Append mode
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.error(f"Failed to create log file handler: {e}")

    return logger

logger = setup_logger() # Initialize with default level

# --- API Retry Decorator ---
def retry_api_call(max_retries: int = API_MAX_RETRIES, delay: float = API_RETRY_DELAY):
    """Decorator to retry CCXT API calls with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"Rate limit exceeded. Retrying in {current_delay:.1f}s... (Attempt {retries + 1}/{max_retries})")
                except ccxt.NetworkError as e:
                    logger.error(f"Network error: {e}. Retrying in {current_delay:.1f}s... (Attempt {retries + 1}/{max_retries})")
                except ccxt.ExchangeError as e:
                    if "Order does not exist" in str(e) or "Order not found" in str(e):
                        logger.warning(f"Order not found (likely filled/cancelled): {e}. Returning None.")
                        return None
                    elif "Insufficient balance" in str(e) or "insufficient margin" in str(e).lower():
                        logger.error(f"Insufficient funds: {e}. Aborting call.")
                        return None  # Or raise if needed
                    else:
                        logger.error(f"Exchange error: {e}. Retrying in {current_delay:.1f}s... (Attempt {retries + 1}/{max_retries})")
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}. Retrying in {current_delay:.1f}s... (Attempt {retries + 1}/{max_retries})")

                time.sleep(current_delay)
                current_delay *= 2
                retries += 1

            logger.error(f"Max retries ({max_retries}) reached for {func.__name__}. Aborting.")
            return None

        return wrapper
    return decorator


# --- Configuration Management ---
def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the configuration dictionary against CONFIG_SECTIONS."""

    validated_config = {}
    for section, schema in CONFIG_SECTIONS.items():
        if schema["required"] and section not in config:
            raise ValueError(f"Missing required section '{section}' in config.")
        elif not schema["required"] and section not in config:
            validated_config[section] = schema.get("default", {}) # Use default if available
            continue

        section_data = config[section]
        if not isinstance(section_data, dict):
            raise ValueError(f"Section '{section}' must be a dictionary.")

        validated_config[section] = section_data # Add section even if empty for consistency

    return validated_config


def load_config(config_file: Path) -> Dict[str, Any]:
    """Loads and validates the YAML configuration file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML dictionary.")
        return validate_config(config)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        logger.error(f"Error loading or validating config file: {e}")
        raise


# --- Scalping Bot Class ---
class ScalpingBot:

    def __init__(self, config_file: Union[str, Path] = CONFIG_FILE_DEFAULT):
        self.config_path = Path(config_file)
        self.config = load_config(self.config_path)

        # Set logging level from config
        log_level = self.config.get("logging_level", "INFO")
        logger.setLevel(getattr(logging, log_level.upper())) # Set level directly
        logger.info(f"Logger level set to {log_level}")


        # Assign config values to attributes (using .get() with defaults for safety)
        self.exchange_id = self.config["exchange"]["exchange_id"]
        self.api_key_env = self.config["exchange"].get("api_key_env", "BYBIT_API_KEY")
        self.api_secret_env = self.config["exchange"].get("api_secret_env", "BYBIT_API_SECRET")

        self.symbol = self.config["trading"]["symbol"]
        self.simulation_mode = self.config["trading"]["simulation_mode"]
        self.entry_order_type = OrderType(self.config["trading"]["entry_order_type"])
        self.limit_order_offset_buy = float(self.config["trading"]["limit_order_offset_buy"])
        self.limit_order_offset_sell = float(self.config["trading"]["limit_order_offset_sell"])
        self.trade_loop_delay = float(self.config["trading"].get("trade_loop_delay_seconds", 10.0))

        self.order_book_depth = int(self.config["order_book"]["depth"])
        if not (1 <= self.order_book_depth <= MAX_ORDER_BOOK_DEPTH):
            raise ValueError(f"order_book_depth must be between 1 and {MAX_ORDER_BOOK_DEPTH}")
        self.imbalance_threshold = float(self.config["order_book"]["imbalance_threshold"])

        self.timeframe = self.config["indicators"].get("timeframe", "1m")
        self.volatility_window = int(self.config["indicators"]["volatility_window"])
        self.volatility_multiplier = float(self.config["indicators"]["volatility_multiplier"])
        self.ema_period = int(self.config["indicators"]["ema_period"])
        self.rsi_period = int(self.config["indicators"]["rsi_period"])

        # ... (rest of indicator parameters)

        self.order_size_percentage = float(self.config["risk_management"]["order_size_percentage"])
        # ... (rest of risk management parameters)


        self.exchange = self._initialize_exchange()
        self.market = self.exchange.market(self.symbol)

        self.open_positions: List[Dict[str, Any]] = []
        self.iteration = 0
        # ... other state variables

        logger.info(f"ScalpingBot initialized for {self.symbol} on {self.exchange_id}. Simulation Mode: {self.simulation_mode}")


    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes the CCXT exchange object."""
        logger.info(f"Initializing exchange: {self.exchange_id}...")
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({"enableRateLimit": True})

            if not self.simulation_mode:
                api_key = os.getenv(self.api_key_env)
                api_secret = os.getenv(self.api_secret_env)
                if api_key and api_secret:
                    exchange.apiKey = api_key
                    exchange.secret = api_secret
                    logger.info("API credentials loaded.")
                else:
                    raise ValueError(f"API key and secret not found in environment variables: {self.api_key_env}, {self.api_secret_env}")

            exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id}.")

            if self.symbol not in exchange.markets:
                raise ValueError(f"Symbol {self.symbol} not available on {self.exchange_id}.")

            # Check required exchange features
            required_features = ["fetchOHLCV", "fetchOrderBook", "createOrder", "fetchBalance"] # Add as needed
            missing_features = [f for f in required_features if not exchange.has.get(f, False)]
            if missing_features:
                raise ValueError(f"Exchange {self.exchange_id} missing required features: {', '.join(missing_features)}")

            return exchange

        except (AttributeError, ValueError, ccxt.ExchangeError) as e:
            logger.error(f"Exchange initialization failed: {e}")
            raise


    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the current market price."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker.get("last") or ticker.get("close") or ticker.get("bid")
            if price is None:
                logger.warning(f"Could not determine market price from ticker: {ticker}")
            return float(price) if price is not None else None
        except ccxt.ExchangeError as e:
            logger.error(f"Error fetching ticker: {e}")
            return None


    # ... (rest of the methods - fetch_order_book, fetch_historical_data, indicator calculations, etc.)

    def run(self):
        """Main trading loop."""
        logger.info(f"Starting trading bot for {self.symbol}...")
        if self.simulation_mode:
            logger.warning("--- RUNNING IN SIMULATION MODE ---")

        while True:
            self.iteration += 1
            logger.info(f"----- Iteration {self.iteration} -----") # Fixed f-string

            try:
                # 1. Fetch Data
                # ...

                # 2. Calculate Indicators
                # ...

                # 3. Manage Open Positions
                # ...

                # 4. Generate Trade Signal
                # ...

                # 5. Place Order
                # ...

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Stopping bot.")
                break
            except ccxt.AuthenticationError as e:
                logger.critical(f"Authentication error: {e}. Exiting.")
                break  # Exit on critical authentication errors
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                # Consider adding a retry mechanism or error handling here.

            time.sleep(self.trade_loop_delay)

        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    try:
        bot = ScalpingBot()
        bot.run()
    except (FileNotFoundError, ValueError, yaml.YAMLError, ccxt.ExchangeError) as e:
        logger.critical(f"Critical initialization error: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled critical error: {e}. Exiting.")
        sys.exit(1)
