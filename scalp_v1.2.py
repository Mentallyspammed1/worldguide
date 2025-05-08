import logging
import logging.handlers
import os
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import ccxt
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Constants ---
CONFIG_FILE_DEFAULT = Path("config.yaml")
LOG_FILE_DEFAULT = Path("scalping_bot.log")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
API_MAX_RETRIES = 3
API_INITIAL_DELAY_SECONDS = 1.0
MIN_HISTORICAL_DATA_BUFFER = 10 # Minimum extra data points beyond strict requirement for indicators

# --- Enums ---
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

# --- Configuration Keys (Used for structure and validation) ---
# Using simple strings for keys is often sufficient and avoids potential Enum complexities
# if the keys themselves don't represent a state machine or complex type.
EXCHANGE_SECTION = "exchange"
TRADING_SECTION = "trading"
ORDER_BOOK_SECTION = "order_book"
INDICATORS_SECTION = "indicators"
RISK_MANAGEMENT_SECTION = "risk_management"
LOGGING_SECTION = "logging" # Added for logging configuration

# --- Setup Logger ---
# Global logger instance, configured by setup_logger
logger = logging.getLogger("ScalpingBot")

def setup_logger(
    level: int = logging.DEBUG,
    log_file: Path = LOG_FILE_DEFAULT,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT
) -> None:
    """
    Configures the global 'ScalpingBot' logger.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Path to the log file.
        max_bytes: Maximum size of the log file before rotation.
        backup_count: Number of backup log files to keep.
    """
    logger.setLevel(level)

    # Prevent adding multiple handlers if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # Filter console output to INFO level by default unless DEBUG is explicitly set
    console_handler.setLevel(logging.INFO if level > logging.DEBUG else logging.DEBUG)
    logger.addHandler(console_handler)

    # Rotating File Handler
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level) # Log everything to file based on the main level
        logger.addHandler(file_handler)
    except IOError as e:
        logger.error(f"Failed to create rotating log file handler for {log_file}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error setting up file logger: {e}")

# Initial basic configuration, will be reconfigured after loading config file
setup_logger()


# --- API Retry Decorator ---
def retry_api_call(
    max_retries: int = API_MAX_RETRIES,
    initial_delay: float = API_INITIAL_DELAY_SECONDS
):
    """
    Decorator to retry CCXT API calls with exponential backoff.

    Handles common transient CCXT errors like RateLimitExceeded, NetworkError,
    and some ExchangeErrors. Logs warnings/errors and returns None after max retries.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            last_exception = None
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    last_exception = e
                    logger.warning(
                        f"Rate limit exceeded for {func.__name__}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                except ccxt.NetworkError as e: # Includes RequestTimeout, DDoSProtection, etc.
                    last_exception = e
                    logger.error(
                        f"Network error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                except ccxt.ExchangeNotAvailable as e:
                    last_exception = e
                    logger.error(
                        f"Exchange not available during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                except ccxt.ExchangeError as e:
                    last_exception = e
                    # Specific handling for potentially non-retryable or expected errors
                    err_str = str(e).lower()
                    if "order not found" in err_str or "order does not exist" in err_str:
                        logger.warning(f"Order not found/does not exist for {func.__name__} (likely filled/cancelled): {e}. Returning None.")
                        return None
                    elif "insufficient balance" in err_str or "insufficient funds" in err_str or "insufficient margin" in err_str:
                         logger.error(f"Insufficient funds for {func.__name__}: {e}. Aborting call.")
                         # Do not retry for insufficient funds, return None or re-raise
                         # raise e # Option to propagate if needed upstream
                         return None # Indicate failure
                    else:
                        # Treat other exchange errors as potentially transient
                        logger.error(
                            f"Exchange error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                            f"(Attempt {retries + 1}/{max_retries + 1})"
                        )
                except Exception as e: # Catch any other unexpected exception
                    last_exception = e
                    logger.exception( # Use exception() to include traceback
                        f"Unexpected error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )

                # Only sleep and increment if we are going to retry
                if retries < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30) # Exponential backoff with a cap (e.g., 30s)
                    retries += 1
                else:
                    break # Exit loop if max retries reached

            logger.error(
                f"Max retries ({max_retries}) reached for API call {func.__name__}. "
                f"Last error: {last_exception}. Aborting call."
            )
            return None # Indicate failure after all retries

        return wrapper
    return decorator


# --- Configuration Validation Schema ---
# Defines the expected structure, types, and constraints for config.yaml.
# This makes validation more declarative and easier to maintain.
CONFIG_SCHEMA = {
    EXCHANGE_SECTION: {
        "required": True,
        "params": {
            "exchange_id": {"type": str, "required": True, "non_empty": True},
            "api_key_env": {"type": str, "required": False, "default": "EXCHANGE_API_KEY"}, # More generic default
            "api_secret_env": {"type": str, "required": False, "default": "EXCHANGE_API_SECRET"}, # More generic default
            "ccxt_options": {"type": dict, "required": False, "default": {"enableRateLimit": True}}, # Allow passing CCXT options
        },
    },
    TRADING_SECTION: {
        "required": True,
        "params": {
            "symbol": {"type": str, "required": True, "non_empty": True},
            "simulation_mode": {"type": bool, "required": True},
            "entry_order_type": {"type": str, "required": True, "allowed": [e.value for e in OrderType]},
            "limit_order_offset_buy_pct": {"type": (int, float), "required": True, "range": (0.0, 1.0)}, # Offset as percentage
            "limit_order_offset_sell_pct": {"type": (int, float), "required": True, "range": (0.0, 1.0)}, # Offset as percentage
            "trade_loop_delay_seconds": {"type": (int, float), "required": False, "default": 10.0, "range": (0.1, float('inf'))},
        },
    },
    ORDER_BOOK_SECTION: {
        "required": True,
        "params": {
            "depth": {"type": int, "required": True, "range": (1, 1000)}, # Practical upper limit
            "imbalance_threshold": {"type": (int, float), "required": True, "range": (0, float('inf'))},
        },
    },
    INDICATORS_SECTION: {
        "required": True,
        "params": {
            "timeframe": {"type": str, "required": True, "non_empty": True}, # Made required, default removed
            "volatility_window": {"type": int, "required": True, "range": (2, float('inf'))}, # Min 2 for std dev
            "volatility_multiplier": {"type": (int, float), "required": True, "range": (0, float('inf'))},
            "ema_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "rsi_period": {"type": int, "required": True, "range": (2, float('inf'))}, # Min 2 for diff
            "macd_short_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "macd_long_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "macd_signal_period": {"type": int, "required": True, "range": (1, float('inf'))},
            "stoch_rsi_period": {"type": int, "required": True, "range": (2, float('inf'))}, # Min 2 for rolling
            "stoch_rsi_smooth_k": {"type": int, "required": False, "default": 3, "range": (1, float('inf'))}, # Added smoothing config
            "stoch_rsi_smooth_d": {"type": int, "required": False, "default": 3, "range": (1, float('inf'))}, # Added smoothing config
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
            "time_based_exit_minutes": {"type": int, "required": False, "default": 0, "range": (0, float('inf'))}, # 0 means disabled
            "trailing_stop_loss_percentage": {"type": (int, float), "required": False, "default": 0.0, "range": (0.0, 1.0)}, # 0 means disabled
            "simulated_balance": {"type": (int, float), "required": False, "default": 10000.0, "range": (0.0, float('inf'))}, # For simulation mode
            "simulated_fee_rate": {"type": (int, float), "required": False, "default": 0.001, "range": (0.0, 1.0)}, # For simulation mode
        },
    },
    LOGGING_SECTION: { # Optional top-level section for logging
        "required": False,
        "params": {
            "level": {
                "type": str,
                "required": False,
                "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO" # Default to INFO for less noise
            },
            "log_file_path": {"type": str, "required": False, "default": str(LOG_FILE_DEFAULT)},
            "log_max_bytes": {"type": int, "required": False, "default": LOG_MAX_BYTES, "range": (1024, float('inf'))},
            "log_backup_count": {"type": int, "required": False, "default": LOG_BACKUP_COUNT, "range": (0, float('inf'))},
        }
    }
}


class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""
    pass


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
            ConfigValidationError: If the configuration is invalid or incomplete.
            AttributeError: If the specified exchange is not supported by CCXT.
            ccxt.AuthenticationError: If API keys are invalid.
            ccxt.ExchangeError: If there's an issue connecting to the exchange.
        """
        self.config_path = Path(config_file)
        self.config: Dict[str, Any] = self._load_and_validate_config() # Raises errors on failure

        # --- Configure Logger based on validated config ---
        log_cfg = self.config.get(LOGGING_SECTION, {})
        log_level_name = log_cfg.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        log_file = Path(log_cfg.get("log_file_path", LOG_FILE_DEFAULT))
        log_max_bytes = log_cfg.get("log_max_bytes", LOG_MAX_BYTES)
        log_backup_count = log_cfg.get("log_backup_count", LOG_BACKUP_COUNT)

        setup_logger(level=log_level, log_file=log_file, max_bytes=log_max_bytes, backup_count=log_backup_count)
        logger.info(f"Logger configured: Level={log_level_name}, File={log_file}")

        # --- Assign validated config values to attributes for easier access ---
        # Using helper method to reduce clutter
        self._assign_config_attributes()

        # --- Initialize Exchange ---
        self.exchange: ccxt.Exchange = self._initialize_exchange() # Raises exceptions on failure
        self.market: Dict[str, Any] = self._load_market_details() # Raises ValueError if symbol invalid

        # --- State Variables ---
        self.open_positions: List[Dict[str, Any]] = [] # Store active positions/orders (needs proper management)
        self.iteration: int = 0
        self.daily_pnl: float = 0.0 # Example metric, needs implementation
        self.last_candle_ts: Optional[int] = None # Track last processed candle timestamp (ms)

        logger.info(f"ScalpingBot initialized for {self.symbol} on {self.exchange_id}. Simulation Mode: {self.simulation_mode}")

    def _assign_config_attributes(self) -> None:
        """Assigns validated configuration values to instance attributes."""
        # Exchange Config
        exch_cfg = self.config[EXCHANGE_SECTION]
        self.exchange_id: str = exch_cfg["exchange_id"]
        self.api_key_env: str = exch_cfg["api_key_env"]
        self.api_secret_env: str = exch_cfg["api_secret_env"]
        self.ccxt_options: dict = exch_cfg["ccxt_options"]

        # Trading Config
        trade_cfg = self.config[TRADING_SECTION]
        self.symbol: str = trade_cfg["symbol"]
        self.simulation_mode: bool = trade_cfg["simulation_mode"]
        self.entry_order_type: OrderType = OrderType(trade_cfg["entry_order_type"])
        self.limit_order_offset_buy_pct: float = float(trade_cfg["limit_order_offset_buy_pct"])
        self.limit_order_offset_sell_pct: float = float(trade_cfg["limit_order_offset_sell_pct"])
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
        self.stoch_rsi_smooth_k: int = ind_cfg["stoch_rsi_smooth_k"]
        self.stoch_rsi_smooth_d: int = ind_cfg["stoch_rsi_smooth_d"]

        # Risk Management Config
        risk_cfg = self.config[RISK_MANAGEMENT_SECTION]
        self.order_size_percentage: float = float(risk_cfg["order_size_percentage"])
        self.stop_loss_pct: float = float(risk_cfg["stop_loss_percentage"])
        self.take_profit_pct: float = float(risk_cfg["take_profit_percentage"])
        self.max_open_positions: int = risk_cfg["max_open_positions"]
        self.time_based_exit_minutes: int = risk_cfg["time_based_exit_minutes"]
        self.trailing_stop_loss_percentage: float = float(risk_cfg["trailing_stop_loss_percentage"])
        self.simulated_balance: float = float(risk_cfg["simulated_balance"])
        self.simulated_fee_rate: float = float(risk_cfg["simulated_fee_rate"])

    def _validate_config_section(self, section_name: str, section_config: Any, section_schema: Dict) -> Dict[str, Any]:
        """Validates a single section of the configuration."""
        if not isinstance(section_config, dict):
            raise ConfigValidationError(f"Section '{section_name}' must be a dictionary.")

        validated_section = {}
        param_schema = section_schema.get("params", {})

        for param, schema in param_schema.items():
            value = section_config.get(param)

            if value is None: # Parameter not present in user config
                if schema.get("required", False):
                    raise ConfigValidationError(f"Missing required parameter '{param}' in section '{section_name}'.")
                elif "default" in schema:
                    validated_section[param] = schema["default"]
                    logger.debug(f"Using default value for '{section_name}.{param}': {schema['default']}")
                else:
                    validated_section[param] = None # Optional parameter without default
                    logger.debug(f"Optional parameter '{section_name}.{param}' not provided.")
                continue # Skip further checks for missing optional params

            # Type check
            expected_type = schema["type"]
            if not isinstance(value, expected_type):
                raise ConfigValidationError(f"Parameter '{param}' in '{section_name}' must be type {expected_type}, got {type(value)}.")

            # Non-empty check for strings
            if schema.get("non_empty", False) and isinstance(value, str) and not value.strip():
                raise ConfigValidationError(f"Parameter '{param}' in '{section_name}' cannot be empty.")

            # Allowed values check
            allowed = schema.get("allowed")
            if allowed and value not in allowed:
                raise ConfigValidationError(f"Parameter '{param}' in '{section_name}' must be one of {allowed}, got '{value}'.")

            # Range check
            value_range = schema.get("range")
            # Ensure value is numeric before range check if type allows int or float
            is_numeric = isinstance(value, (int, float))
            if value_range and is_numeric and not (value_range[0] <= value <= value_range[1]):
                raise ConfigValidationError(f"Parameter '{param}' in '{section_name}' must be between {value_range[0]} and {value_range[1]}, got {value}.")
            elif value_range and not is_numeric:
                 # This case should ideally be caught by type check, but added as safeguard
                 logger.warning(f"Range check skipped for non-numeric parameter '{param}' in '{section_name}'.")


            validated_section[param] = value # Assign validated value

        # Check for unexpected parameters
        extra_params = set(section_config.keys()) - set(param_schema.keys())
        if extra_params:
             logger.warning(f"Ignoring unexpected parameters in section '{section_name}': {', '.join(extra_params)}")

        # Custom validations for the section
        custom_validations = section_schema.get("custom_validations", [])
        for validation_func in custom_validations:
            try:
                result = validation_func(validated_section)
                if isinstance(result, str): # Expect error message string on failure
                    raise ConfigValidationError(f"Custom validation failed for section '{section_name}': {result}")
            except KeyError as e:
                 raise ConfigValidationError(f"Custom validation for '{section_name}' failed: Missing parameter {e} needed for validation.")
            except Exception as e:
                 raise ConfigValidationError(f"Error during custom validation for '{section_name}': {e}")


        return validated_section

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Loads and validates the configuration file using CONFIG_SCHEMA."""
        logger.info(f"Loading configuration from {self.config_path}...")
        if not self.config_path.is_file():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config_raw = yaml.safe_load(f)
            if not isinstance(config_raw, dict):
                raise ConfigValidationError("Config file is not a valid YAML dictionary.")
            logger.info("Configuration file loaded successfully.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file: {e}")
            raise # Re-raise the original error
        except Exception as e:
            logger.error(f"Error reading configuration file {self.config_path}: {e}")
            raise

        logger.info("Validating configuration...")
        validated_config = {}
        all_sections = set(config_raw.keys())
        defined_sections = set(CONFIG_SCHEMA.keys())

        for section_name, section_schema in CONFIG_SCHEMA.items():
            if section_name not in config_raw:
                if section_schema.get("required", False):
                    raise ConfigValidationError(f"Missing required section '{section_name}' in config file.")
                else:
                    logger.debug(f"Optional section '{section_name}' not found, skipping validation.")
                    # Add default structure if section is optional but has params with defaults
                    validated_config[section_name] = self._validate_config_section(section_name, {}, section_schema)
            else:
                 validated_config[section_name] = self._validate_config_section(
                     section_name, config_raw[section_name], section_schema
                 )

        # Check for unexpected top-level sections
        extra_sections = all_sections - defined_sections
        if extra_sections:
            logger.warning(f"Ignoring unexpected top-level sections in config file: {', '.join(extra_sections)}")

        logger.info("Configuration validated successfully.")
        return validated_config

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and returns the CCXT exchange instance."""
        logger.info(f"Initializing exchange: {self.exchange_id.upper()}...")
        try:
            exchange_class: Type[ccxt.Exchange] = getattr(ccxt, self.exchange_id)
        except AttributeError:
            logger.error(f"Exchange ID '{self.exchange_id}' is not a valid CCXT exchange.")
            raise AttributeError(f"Exchange ID '{self.exchange_id}' is not a valid CCXT exchange.")

        api_key = os.getenv(self.api_key_env)
        api_secret = os.getenv(self.api_secret_env)

        exchange_config = self.ccxt_options.copy() # Start with options from config

        if not self.simulation_mode:
            if api_key and api_secret:
                exchange_config["apiKey"] = api_key
                exchange_config["secret"] = api_secret
                logger.info(f"API credentials loaded from environment variables ({self.api_key_env}, {self.api_secret_env}).")
            else:
                logger.warning(
                    f"Running in LIVE mode but API Key ({self.api_key_env}) or Secret ({self.api_secret_env}) "
                    "environment variables not set. Trading operations will likely fail."
                )
        else:
            logger.info("Running in SIMULATION mode. No API credentials needed or used.")

        try:
            exchange = exchange_class(exchange_config)
            # Set sandbox mode if available and in simulation (some exchanges support this)
            if self.simulation_mode and exchange.has.get('sandbox'):
                 exchange.set_sandbox_mode(True)
                 logger.info("Enabled exchange sandbox mode for simulation.")

            logger.debug("Loading markets...")
            exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id.upper()} successfully.")

            # Check required capabilities after loading markets
            required_capabilities = {
                'fetchOHLCV': exchange.has['fetchOHLCV'],
                'fetchOrderBook': exchange.has['fetchOrderBook'],
                'fetchTicker': exchange.has['fetchTicker'],
                'createOrder': exchange.has['createOrder'] if not self.simulation_mode else True, # Assume simulation can "create" orders
                'fetchBalance': exchange.has['fetchBalance'] if not self.simulation_mode else True, # Assume simulation can "fetch" balance
            }
            for cap, available in required_capabilities.items():
                if not available:
                    # Log error if critical capability missing in live mode
                    log_func = logger.error if cap in ['createOrder', 'fetchBalance'] and not self.simulation_mode else logger.warning
                    log_func(f"Exchange {self.exchange_id} does not support required capability: {cap}")
                    if log_func == logger.error:
                         raise ccxt.NotSupported(f"Exchange {self.exchange_id} missing critical capability: {cap}")


            return exchange

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication Error with {self.exchange_id}: {e}. Check API keys/permissions.")
            raise
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange Not Available: {self.exchange_id} might be down or unreachable. {e}")
            raise
        except ccxt.NetworkError as e:
             logger.error(f"Network error during exchange initialization: {e}")
             raise
        except ccxt.ExchangeError as e: # Catch other specific CCXT exchange errors
             logger.error(f"Exchange error during initialization: {e}")
             raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during exchange initialization: {e}")
            raise # Re-raise unexpected errors

    def _load_market_details(self) -> Dict[str, Any]:
        """Loads and validates market details for the configured symbol."""
        logger.debug(f"Loading market details for symbol: {self.symbol}")
        try:
            market = self.exchange.market(self.symbol)
            if market is None:
                 raise ValueError(f"Market details not found for symbol '{self.symbol}'")

             # Log key market details
            logger.info(f"Market details for {self.symbol}: Base={market.get('base')}, Quote={market.get('quote')}")
            logger.debug(f"Precision: Amount={market.get('precision', {}).get('amount')}, Price={market.get('precision', {}).get('price')}")
            logger.debug(f"Limits: Amount Min={market.get('limits', {}).get('amount', {}).get('min')}, Cost Min={market.get('limits', {}).get('cost', {}).get('min')}")

            # Basic validation of critical market info
            if not market.get('base') or not market.get('quote'):
                raise ValueError(f"Base or Quote currency missing in market details for {self.symbol}")
            if market.get('precision') is None or market.get('limits') is None:
                 logger.warning(f"Precision or Limits information might be incomplete for {self.symbol}")

            return market

        except ValueError as e: # Catch specific error from market() if symbol not found by CCXT
             logger.error(f"Symbol '{self.symbol}' not found or invalid on {self.exchange_id}.")
             available_symbols = list(self.exchange.markets.keys())
             logger.info(f"Available symbols sample ({len(available_symbols)} total): {available_symbols[:20]}...")
             raise ValueError(f"Symbol '{self.symbol}' not available on exchange '{self.exchange_id}'") from e
        except Exception as e:
             logger.exception(f"Unexpected error loading market details for {self.symbol}: {e}")
             raise


    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the last traded price for the symbol."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        if not ticker:
            logger.warning(f"fetch_ticker returned empty or None for {self.symbol}")
            return None

        # Prefer 'last', fallback to 'close', then 'ask'/'bid' midpoint as last resort
        price = ticker.get("last") or ticker.get("close")
        if price is None:
            ask = ticker.get("ask")
            bid = ticker.get("bid")
            if ask is not None and bid is not None and ask > 0 and bid > 0:
                 price = (ask + bid) / 2.0
                 logger.debug(f"Using midpoint price: {price:.{self.market.get('precision', {}).get('price', 2)}f}")
            else:
                 logger.warning(f"Market price ('last', 'close', 'ask'/'bid') unavailable in ticker for {self.symbol}. Ticker: {ticker}")
                 return None

        try:
            price_float = float(price)
            logger.debug(f"Fetched market price: {price_float:.{self.market.get('precision', {}).get('price', 2)}f}")
            return price_float
        except (ValueError, TypeError) as e:
             logger.error(f"Failed to convert fetched price '{price}' to float: {e}")
             return None


    @retry_api_call()
    def fetch_order_book(self) -> Optional[float]:
        """Fetches order book and calculates the bid/ask volume imbalance ratio."""
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)

        if not orderbook or not isinstance(orderbook, dict):
             logger.warning(f"fetch_order_book returned invalid data for {self.symbol}: {orderbook}")
             return None

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            logger.warning(f"Order book data missing bids or asks for {self.symbol}. Bids: {len(bids)}, Asks: {len(asks)}")
            return None # Cannot calculate imbalance without both sides

        # Calculate total volume within the specified depth
        # Order book format: [[price, amount], [price, amount], ...]
        try:
            bid_volume = sum(bid[1] for bid in bids if len(bid) > 1 and isinstance(bid[1], (int, float)))
            ask_volume = sum(ask[1] for ask in asks if len(ask) > 1 and isinstance(ask[1], (int, float)))
        except (TypeError, IndexError) as e:
             logger.error(f"Error processing order book data for volume calculation: {e}. Bids: {bids}, Asks: {asks}")
             return None

        if ask_volume <= 0:
            # Handle division by zero: Infinite imbalance if bids exist, neutral (1.0) otherwise
            imbalance_ratio = float("inf") if bid_volume > 0 else 1.0
            logger.debug(f"Ask volume is zero or negative ({ask_volume}). Setting imbalance ratio to {imbalance_ratio}.")
        else:
            imbalance_ratio = bid_volume / ask_volume

        logger.debug(f"Order Book - Top {self.order_book_depth} levels: "
                     f"Bid Vol: {bid_volume:.4f}, Ask Vol: {ask_volume:.4f}, "
                     f"Bid/Ask Ratio: {imbalance_ratio:.2f}")
        return imbalance_ratio


    @retry_api_call()
    def fetch_historical_data(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data for the configured symbol and timeframe.

        Args:
            limit: The number of candles to fetch. If None, calculates based on
                   indicator requirements plus a buffer.

        Returns:
            A pandas DataFrame with OHLCV data indexed by timestamp, or None on failure.
        """
        # Determine the minimum number of candles required for all indicators
        # Add 1 for calculations involving differences or shifts
        min_required = max(
            self.volatility_window + 1,
            self.ema_period, # EMA doesn't strictly need +1 but more data is better
            self.rsi_period + 1, # RSI needs period + 1 for diff
            self.macd_long_period + self.macd_signal_period, # MACD needs long + signal for full calculation
            self.rsi_period + self.stoch_rsi_period + self.stoch_rsi_smooth_k + self.stoch_rsi_smooth_d, # Stoch RSI chain
            MIN_HISTORICAL_DATA_BUFFER # Ensure a baseline minimum
        )

        # Fetch slightly more than strictly needed for stability and lookback calculations
        fetch_limit = limit if limit is not None else min_required + MIN_HISTORICAL_DATA_BUFFER

        logger.debug(f"Fetching {fetch_limit} historical {self.timeframe} candles for {self.symbol} (min required: {min_required})...")

        if not self.exchange.has['fetchOHLCV']:
             logger.error(f"Exchange {self.exchange_id} does not support fetchOHLCV. Cannot fetch historical data.")
             return None

        try:
            # Fetch OHLCV data using CCXT
            # CCXT returns: [timestamp, open, high, low, close, volume]
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=fetch_limit)

            if not ohlcv or len(ohlcv) < 2: # Need at least 2 points for most calculations
                logger.warning(f"Insufficient or no historical OHLCV data returned for {self.symbol} with timeframe {self.timeframe}. Got {len(ohlcv) if ohlcv else 0} candles.")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Convert timestamp to datetime objects (UTC) and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaN in critical columns (open, high, low, close)
            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} rows with NaN values from OHLCV data.")

            # Drop potential duplicates just in case API returns overlapping data
            initial_len = len(df)
            df = df[~df.index.duplicated(keep='first')]
            if len(df) < initial_len:
                 logger.warning(f"Dropped {initial_len - len(df)} duplicate timestamp rows from OHLCV data.")


            # Final check for sufficient data length after cleaning
            if len(df) < min_required:
                logger.error(f"Insufficient historical data after cleaning. Requested ~{fetch_limit}, got {len(df)}, minimum required {min_required}. Cannot proceed with calculations.")
                return None

            logger.debug(f"Historical data fetched and processed successfully. Shape: {df.shape}, Time Range: {df.index.min()} to {df.index.max()}")
            return df

        except ccxt.NetworkError as e:
             logger.error(f"Network error fetching OHLCV data: {e}")
             return None # Caught by retry, but handle here too
        except ccxt.ExchangeError as e:
             logger.error(f"Exchange error fetching OHLCV data: {e}")
             return None
        except Exception as e:
            logger.exception(f"An unexpected error occurred while fetching or processing historical data: {e}")
            return None

    # --- Indicator Calculations ---

    def _calculate_indicator(
        self,
        data: Optional[pd.Series],
        required_length: int,
        calc_func: callable,
        name: str,
        **kwargs
    ) -> Optional[Any]:
        """
        Helper to safely calculate an indicator on a pandas Series.

        Checks for None input, sufficient data length, executes the calculation,
        extracts the last value if a Series is returned, checks for NaN, and logs.

        Args:
            data: The pandas Series to calculate on (e.g., close prices).
            required_length: Minimum number of data points needed.
            calc_func: The function performing the indicator calculation.
                       It should accept the Series as the first argument and any kwargs.
            name: The name of the indicator for logging.
            **kwargs: Additional keyword arguments passed to calc_func.

        Returns:
            The calculated indicator value (usually the last point), or None if calculation fails.
        """
        if data is None:
            logger.debug(f"Input data is None for {name} calculation.")
            return None
        if len(data) < required_length:
            logger.warning(f"Insufficient data for {name} calculation (need {required_length}, got {len(data)})")
            return None

        try:
            result = calc_func(data, **kwargs)

            # Extract the last value if the result is a Series or DataFrame column
            if isinstance(result, pd.Series):
                if result.empty:
                     logger.warning(f"{name} calculation resulted in an empty Series.")
                     return None
                result_val = result.iloc[-1]
            else:
                # Assume scalar result or handle other types if necessary
                result_val = result

            # Check for NaN or infinite values
            if pd.isna(result_val) or np.isinf(result_val):
                logger.warning(f"{name} calculation resulted in NaN or infinite value.")
                return None

            # Log scalar results with formatting, handle tuples/other types separately
            log_msg = f"Calculated {name}: "
            if isinstance(result_val, (float, np.floating)):
                 log_msg += f"{result_val:.4f}" # Adjust precision as needed
            elif isinstance(result_val, tuple):
                 log_msg += ", ".join([f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in result_val])
            else:
                 log_msg += str(result_val)
            logger.debug(log_msg)

            return result_val
        except Exception as e:
            logger.error(f"Error calculating {name}: {e}", exc_info=False) # Set exc_info=False to avoid full traceback for common calc errors
            return None


    def calculate_volatility(self, data: pd.DataFrame) -> Optional[float]:
        """Calculates the rolling log return standard deviation (volatility)."""
        if data is None or 'close' not in data.columns: return None
        required = self.volatility_window + 1 # Need one previous point for shift/diff
        return self._calculate_indicator(
            data['close'], required,
            lambda s: np.log(s / s.shift(1)).rolling(window=self.volatility_window).std(),
            f"Volatility (window {self.volatility_window})"
        )

    def calculate_ema(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """Calculates the Exponential Moving Average (EMA)."""
        if data is None or 'close' not in data.columns: return None
        return self._calculate_indicator(
            data['close'], period, # EMA needs at least 'period' points to stabilize
            lambda s: s.ewm(span=period, adjust=False).mean(),
            f"EMA (period {period})"
        )

    def calculate_rsi(self, data: pd.DataFrame) -> Optional[float]:
        """Calculates the Relative Strength Index (RSI)."""
        if data is None or 'close' not in data.columns: return None
        required = self.rsi_period + 1 # For diff() and initial smoothing

        def _rsi_calc(series: pd.Series, period: int):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).fillna(0.0)
            loss = -delta.where(delta < 0, 0.0).fillna(0.0)

            # Use simple moving average for the first period, then EWMA
            # This aligns better with common definitions but requires more data initially.
            # Alternatively, use EWMA directly (adjust=False) as before, which is simpler.
            # Sticking with EWMA for simplicity here.
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

            rs = avg_gain / avg_loss.replace(0, 1e-10) # Avoid division by zero
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

        return self._calculate_indicator(
            data['close'], required, _rsi_calc, f"RSI (period {self.rsi_period})",
            period=self.rsi_period # Pass period to the calc function
        )

    def calculate_macd(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculates the Moving Average Convergence Divergence (MACD).

        Returns:
            A tuple containing (MACD Line, Signal Line, Histogram), or (None, None, None) on failure.
        """
        if data is None or 'close' not in data.columns: return None, None, None
        # Required length is roughly long_period + signal_period for stable values
        required = self.macd_long_period + self.macd_signal_period

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

            # Get the last values
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]
            hist_val = histogram.iloc[-1]

            if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
                logger.warning("MACD calculation resulted in NaN values.")
                return None, None, None

            logger.debug(f"MACD: {macd_val:.4f}, Signal: {signal_val:.4f}, Histogram: {hist_val:.4f}")
            return macd_val, signal_val, hist_val
        except Exception as e:
             logger.error(f"Error calculating MACD: {e}", exc_info=False)
             return None, None, None


    def calculate_stoch_rsi(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates the Stochastic RSI.

        Returns:
            A tuple containing (%K, %D), or (None, None) on failure.
        """
        if data is None or 'close' not in data.columns: return None, None
        # Stoch RSI needs RSI calculation + rolling window + k smoothing + d smoothing
        required = self.rsi_period + self.stoch_rsi_period + self.stoch_rsi_smooth_k + self.stoch_rsi_smooth_d

        close_series = data['close']
        if len(close_series) < required:
             logger.warning(f"Insufficient data for Stoch RSI calculation (need ~{required}, got {len(close_series)})")
             return None, None

        try:
            # 1. Calculate RSI first
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0.0).fillna(0.0)
            loss = -delta.where(delta < 0, 0.0).fillna(0.0)
            avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            rsi_series.dropna(inplace=True) # Drop initial NaNs from RSI calculation

            if len(rsi_series) < self.stoch_rsi_period + self.stoch_rsi_smooth_k + self.stoch_rsi_smooth_d:
                 logger.warning(f"Insufficient RSI values for Stoch RSI rolling/smoothing ({len(rsi_series)} available)")
                 return None, None

            # 2. Calculate Stoch RSI %K (raw)
            min_rsi = rsi_series.rolling(window=self.stoch_rsi_period).min()
            max_rsi = rsi_series.rolling(window=self.stoch_rsi_period).max()
            stoch_rsi_k_raw = 100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-10) # Avoid div by zero

            # 3. Smooth %K and %D
            k = stoch_rsi_k_raw.rolling(window=self.stoch_rsi_smooth_k).mean()
            d = k.rolling(window=self.stoch_rsi_smooth_d).mean()

            # Get the last values
            k_val = k.iloc[-1]
            d_val = d.iloc[-1]

            if pd.isna(k_val) or pd.isna(d_val):
                logger.warning("Stochastic RSI calculation resulted in NaN.")
                return None, None

            logger.debug(f"Calculated Stochastic RSI (period {self.stoch_rsi_period}, K={self.stoch_rsi_smooth_k}, D={self.stoch_rsi_smooth_d}) - K: {k_val:.2f}, D: {d_val:.2f}")
            return k_val, d_val
        except Exception as e:
             logger.error(f"Error calculating Stochastic RSI: {e}", exc_info=False)
             return None, None

    # --- Balance and Order Sizing ---

    @retry_api_call()
    def fetch_balance(self) -> Optional[float]:
        """Fetches the free balance for the quote currency."""
        if self.simulation_mode:
            # In simulation mode, return the configured simulated balance
            logger.debug(f"Simulation mode: Using simulated balance of {self.simulated_balance}.")
            return self.simulated_balance

        # Live mode: Fetch from exchange
        try:
            balance_info = self.exchange.fetch_balance()
            quote_currency = self.market.get("quote")
            if not quote_currency:
                 logger.error("Could not determine quote currency from market data.")
                 return None

            # Use 'free' balance for placing new orders
            free_balance = balance_info.get("free", {}).get(quote_currency)

            if free_balance is None:
                 logger.warning(f"Could not find free balance for quote currency '{quote_currency}' in balance response. Response: {balance_info}")
                 # Fallback: Check 'total' or return 0? Returning 0 is safer.
                 total_balance = balance_info.get("total", {}).get(quote_currency, 0.0)
                 logger.warning(f"Free balance unavailable, total balance is {total_balance}. Returning 0.0 as free balance.")
                 free_balance = 0.0 # Safer to assume 0 free if not explicitly found

            logger.debug(f"Fetched free balance for {quote_currency}: {free_balance}")
            return float(free_balance)

        except ccxt.AuthenticationError:
             logger.error("Authentication required to fetch balance. Check API keys/permissions.")
             return None # Don't retry auth errors automatically here
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
             # These should be handled by the retry decorator, but log specific message if it gets here
             logger.error(f"Error fetching balance after retries: {e}")
             return None
        except Exception as e:
            logger.exception(f"Unexpected error retrieving balance for {self.market.get('quote', 'quote currency')}: {e}")
            return None

    def _get_market_precision(self) -> Tuple[Optional[int], Optional[int]]:
        """Safely retrieves amount and price precision digits from market info."""
        amount_precision_raw = self.market.get("precision", {}).get("amount")
        price_precision_raw = self.market.get("precision", {}).get("price")

        # CCXT precision can be integer digits or decimal places (e.g., 0.001 -> 3)
        # We need the number of decimal places for formatting.
        def parse_precision(precision_val):
            if precision_val is None: return None
            if isinstance(precision_val, int): return precision_val # Assume it's decimal places if int
            try: # Handle cases like 1e-8 or 0.0001
                precision_float = float(precision_val)
                if 0 < precision_float < 1:
                    # Calculate decimal places from float like 0.001
                    return abs(int(np.log10(precision_float)))
                elif precision_float == 1:
                     return 0 # Integer precision
                else: # Handle cases like 10, 100 (meaning round to nearest 10, 100) - less common for crypto
                     logger.warning(f"Unhandled precision format: {precision_val}. Assuming 0 decimal places.")
                     return 0
            except (ValueError, TypeError):
                logger.error(f"Could not parse precision value: {precision_val}")
                return None

        amount_precision = parse_precision(amount_precision_raw)
        price_precision = parse_precision(price_precision_raw)
        return amount_precision, price_precision

    def _format_value(self, value: float, precision_type: str) -> Optional[str]:
         """Formats amount or price according to market precision using ccxt helper."""
         if value is None: return None
         try:
             if precision_type == 'amount':
                 return self.exchange.amount_to_precision(self.symbol, value)
             elif precision_type == 'price':
                 return self.exchange.price_to_precision(self.symbol, value)
             else:
                 logger.error(f"Invalid precision type '{precision_type}' for formatting.")
                 return None
         except Exception as e:
              logger.error(f"Error formatting {precision_type} ({value}) using exchange precision: {e}")
              return None # Return None if formatting fails

    def calculate_order_size(self, price: float, volatility: Optional[float]) -> Optional[float]:
        """
        Calculates the order size in the base currency, considering risk percentage,
        volatility adjustment (optional), and exchange minimums/precision.

        Args:
            price: The current market price (used for conversion and min cost check).
            volatility: The calculated volatility (optional, used for size adjustment).

        Returns:
            The calculated order size in base currency, adjusted for precision and minimums,
            or None if calculation fails or size is too small.
        """
        if price <= 0:
            logger.error(f"Invalid price ({price}) for order size calculation.")
            return None

        balance = self.fetch_balance()
        if balance is None or balance <= 0:
            logger.warning(f"Cannot calculate order size: Balance is {balance}.")
            return None

        # 1. Calculate base size based on risk percentage of balance (in quote currency)
        base_size_quote = balance * self.order_size_percentage
        if base_size_quote <= 0:
             logger.warning(f"Initial order size based on balance ({balance:.4f}) and risk percentage ({self.order_size_percentage:.2%}) is zero or negative.")
             return None
        logger.debug(f"Base order size (quote): {base_size_quote:.4f} {self.market['quote']} ({self.order_size_percentage:.2%} of {balance:.2f})")

        # 2. Adjust size based on volatility (optional)
        adjusted_size_quote = base_size_quote
        if volatility is not None and volatility > 1e-9 and self.volatility_multiplier > 0:
            # Reduce size more in higher volatility; adjustment factor <= 1
            # Simple inverse relationship: size = base_size / (1 + vol * multiplier)
            adjustment_factor = 1.0 / (1.0 + (volatility * self.volatility_multiplier))
            adjusted_size_quote = base_size_quote * adjustment_factor
            logger.debug(f"Volatility adjustment factor: {adjustment_factor:.4f} (Volatility: {volatility:.5f}, Multiplier: {self.volatility_multiplier})")
            logger.debug(f"Adjusted order size (quote): {adjusted_size_quote:.4f} {self.market['quote']}")
        else:
            logger.debug("No volatility adjustment applied (Volatility N/A, zero, or multiplier is 0).")

        # 3. Convert quote size to base size
        if price <= 0: # Double check price just before division
             logger.error("Price is zero or negative before converting to base size.")
             return None
        calculated_size_base = adjusted_size_quote / price
        logger.debug(f"Calculated base size before constraints: {calculated_size_base:.8f} {self.market['base']}")

        # 4. Apply exchange constraints (minimums and precision)
        limits = self.market.get("limits", {})
        min_amount = limits.get("amount", {}).get("min")
        min_cost = limits.get("cost", {}).get("min")
        amount_precision_digits, _ = self._get_market_precision()

        final_size_base = calculated_size_base

        # Check minimum cost first (if available)
        if min_cost is not None and adjusted_size_quote < min_cost:
            logger.warning(f"Calculated order cost ({adjusted_size_quote:.4f} {self.market['quote']}) is below exchange minimum cost ({min_cost}).")
            # Option 1: Increase size to meet min_cost if it doesn't exceed original risk budget
            if min_cost <= base_size_quote:
                 logger.info(f"Attempting to adjust size to meet minimum cost ({min_cost} {self.market['quote']}).")
                 final_size_base = min_cost / price
                 logger.debug(f"Adjusted base size to meet min cost: {final_size_base:.8f} {self.market['base']}")
            # Option 2: Fail the trade
            else:
                 logger.error(f"Minimum order cost ({min_cost} {self.market['quote']}) exceeds allocated risk capital ({base_size_quote:.2f}). Cannot place trade.")
                 return None

        # Apply amount precision *before* checking min amount
        if amount_precision_digits is not None:
            try:
                # Use floor division or truncation logic appropriate for the exchange/asset
                # CCXT's amount_to_precision usually handles this correctly (truncates)
                precise_size_base_str = self._format_value(final_size_base, 'amount')
                if precise_size_base_str is None: raise ValueError("Formatting returned None")
                precise_size_base = float(precise_size_base_str)

                if precise_size_base <= 1e-12: # Check for effective zero after precision
                    logger.warning(f"Order size became zero or negligible ({precise_size_base}) after applying amount precision. Initial base size: {final_size_base:.8f}")
                    return None
                if precise_size_base != final_size_base:
                     logger.debug(f"Applied amount precision ({amount_precision_digits} decimals): {precise_size_base:.8f} (from {final_size_base:.8f})")
                final_size_base = precise_size_base
            except Exception as e:
                 logger.error(f"Failed to apply amount precision: {e}. Cannot determine valid order size.")
                 return None # Safer to fail if precision cannot be applied

        # Check minimum amount *after* applying precision
        if min_amount is not None and final_size_base < min_amount:
            logger.warning(f"Calculated order size {final_size_base:.8f} is below exchange minimum amount {min_amount}.")
            # Option 1: Increase to minimum amount if allowed by risk and cost
            min_amount_cost = min_amount * price
            if min_amount_cost <= base_size_quote: # Check against initial risk budget
                logger.info(f"Adjusting order size to exchange minimum amount: {min_amount} {self.market['base']}")
                final_size_base = min_amount
                # Re-apply precision just in case min_amount itself needs it (unlikely but possible)
                if amount_precision_digits is not None:
                    precise_min_amount_str = self._format_value(final_size_base, 'amount')
                    if precise_min_amount_str:
                         final_size_base = float(precise_min_amount_str)
                    else:
                         logger.error("Failed to re-apply precision to minimum amount. Aborting.")
                         return None
            # Option 2: Fail the trade
            else:
                logger.error(f"Minimum order amount ({min_amount} {self.market['base']} = ~{min_amount_cost:.2f} {self.market['quote']}) "
                               f"cost exceeds the allocated risk capital ({base_size_quote:.2f}). Skipping trade.")
                return None

        # Final check for zero or negative size after all adjustments
        if final_size_base <= 1e-12: # Use a small threshold
             logger.error(f"Final calculated order size is zero or negative ({final_size_base}). Skipping trade.")
             return None

        final_cost = final_size_base * price
        logger.info(
            f"Final Calculated Order Size: {final_size_base:.8f} {self.market['base']} "
            f"(Quote Value: ~{final_cost:.{self.market.get('precision', {}).get('price', 2)}f} {self.market['quote']})"
        )
        return final_size_base


    # --- Trade Execution Logic ---

    def compute_trade_signal_score(self, price: float, indicators: dict, orderbook_imbalance: Optional[float]) -> Tuple[int, List[str]]:
        """
        Computes a simple score based on indicator signals and order book imbalance.
        Higher positive score suggests BUY, higher negative score suggests SELL.

        Args:
            price: Current market price.
            indicators: Dictionary containing calculated indicator values.
            orderbook_imbalance: Calculated bid/ask volume ratio.

        Returns:
            A tuple containing (score: int, reasons: List[str]).
        """
        score = 0
        reasons = []

        # --- Extract Indicator Values ---
        # Use .get() to handle potential None values gracefully
        ema = indicators.get("ema")
        rsi = indicators.get("rsi")
        macd_line = indicators.get("macd_line")
        macd_signal = indicators.get("macd_signal")
        # macd_hist = indicators.get("macd_hist") # Histogram available if needed
        stoch_rsi_k = indicators.get("stoch_rsi_k")
        stoch_rsi_d = indicators.get("stoch_rsi_d")

        # --- Define Thresholds (Consider making these configurable) ---
        rsi_oversold = 30
        rsi_overbought = 70
        stoch_rsi_oversold = 20
        stoch_rsi_overbought = 80

        # --- Order Book Imbalance ---
        if orderbook_imbalance is not None:
            # Ensure imbalance_threshold is positive to avoid division by zero if used for sell signal
            sell_imbalance_threshold = (1.0 / self.imbalance_threshold) if self.imbalance_threshold > 1e-9 else float('-inf')

            if orderbook_imbalance > self.imbalance_threshold:
                score += 1
                reasons.append(f"OB Imbalance: Strong bid pressure (Ratio: {orderbook_imbalance:.2f} > {self.imbalance_threshold:.2f})")
            elif orderbook_imbalance < sell_imbalance_threshold:
                score -= 1
                reasons.append(f"OB Imbalance: Strong ask pressure (Ratio: {orderbook_imbalance:.2f} < {sell_imbalance_threshold:.2f})")
            else:
                reasons.append(f"OB Imbalance: Neutral pressure (Ratio: {orderbook_imbalance:.2f})")
        else:
             reasons.append("OB Imbalance: N/A")

        # --- EMA Trend ---
        if ema is not None:
            if price > ema:
                score += 1
                reasons.append(f"Trend: Price > EMA ({price:.{self.market.get('precision', {}).get('price', 2)}f} > {ema:.{self.market.get('precision', {}).get('price', 2)}f}) (Bullish)")
            elif price < ema:
                score -= 1
                reasons.append(f"Trend: Price < EMA ({price:.{self.market.get('precision', {}).get('price', 2)}f} < {ema:.{self.market.get('precision', {}).get('price', 2)}f}) (Bearish)")
            else:
                reasons.append(f"Trend: Price == EMA ({price:.{self.market.get('precision', {}).get('price', 2)}f})")
        else:
             reasons.append("Trend: EMA N/A")

        # --- RSI Overbought/Oversold ---
        if rsi is not None:
            if rsi < rsi_oversold:
                score += 1 # Potential buy signal (oversold)
                reasons.append(f"Momentum: RSI < {rsi_oversold} ({rsi:.2f}) (Oversold)")
            elif rsi > rsi_overbought:
                score -= 1 # Potential sell signal (overbought)
                reasons.append(f"Momentum: RSI > {rsi_overbought} ({rsi:.2f}) (Overbought)")
            else:
                reasons.append(f"Momentum: RSI Neutral ({rsi:.2f})")
        else:
             reasons.append("Momentum: RSI N/A")

        # --- MACD Crossover ---
        if macd_line is not None and macd_signal is not None:
            # Could add check for previous state to confirm crossover
            if macd_line > macd_signal:
                score += 1 # Bullish crossover/state
                reasons.append(f"Momentum: MACD > Signal ({macd_line:.4f} > {macd_signal:.4f}) (Bullish)")
            elif macd_line < macd_signal:
                score -= 1 # Bearish crossover/state
                reasons.append(f"Momentum: MACD < Signal ({macd_line:.4f} < {macd_signal:.4f}) (Bearish)")
            else:
                reasons.append(f"Momentum: MACD == Signal ({macd_line:.4f})")
        else:
             reasons.append("Momentum: MACD N/A")

        # --- Stochastic RSI Overbought/Oversold ---
        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            # Simple check: both lines in extreme zones
            if stoch_rsi_k < stoch_rsi_oversold and stoch_rsi_d < stoch_rsi_oversold:
                score += 1 # Oversold
                reasons.append(f"Momentum: Stoch RSI < {stoch_rsi_oversold} (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (Oversold)")
            elif stoch_rsi_k > stoch_rsi_overbought and stoch_rsi_d > stoch_rsi_overbought:
                score -= 1 # Overbought
                reasons.append(f"Momentum: Stoch RSI > {stoch_rsi_overbought} (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (Overbought)")
            # Optional: Add crossover logic (e.g., K crossing above D in oversold zone)
            # elif stoch_rsi_k > stoch_rsi_d and stoch_rsi_k < 50: # Example bullish crossover
            #     score += 1
            #     reasons.append(f"Momentum: Stoch RSI K crossed above D (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f})")
            else:
                reasons.append(f"Momentum: Stoch RSI Neutral (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f})")
        else:
             reasons.append("Momentum: Stoch RSI N/A")

        logger.debug(f"Signal Score: {score}, Reasons: {'; '.join(reasons)}")
        return score, reasons


    @retry_api_call(max_retries=1) # Only retry order placement once by default, as market conditions change fast
    def place_order(
        self,
        side: OrderSide,
        order_size: float,
        order_type: OrderType,
        price: Optional[float] = None, # Required for limit orders
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Places a trade order (simulated or live) with optional SL/TP.

        Args:
            side: OrderSide.BUY or OrderSide.SELL.
            order_size: The amount of base currency to trade.
            order_type: OrderType.MARKET or OrderType.LIMIT.
            price: The price for LIMIT orders.
            stop_loss_price: Optional stop-loss trigger price.
            take_profit_price: Optional take-profit trigger price.

        Returns:
            The order dictionary returned by CCXT (or a simulated one), or None on failure.
        """

        # --- Input Validation ---
        if not isinstance(side, OrderSide):
            logger.error(f"Invalid order side type: {type(side)}. Use OrderSide enum.")
            return None
        if not isinstance(order_type, OrderType):
             logger.error(f"Invalid order type type: {type(order_type)}. Use OrderType enum.")
             return None
        if order_type == OrderType.LIMIT and price is None:
            logger.error(f"Price is required for {OrderType.LIMIT.value} orders.")
            return None
        if order_size <= 0:
            logger.error(f"Order size must be positive, got {order_size}.")
            return None
        if price is not None and price <= 0:
             logger.error(f"Order price must be positive for limit orders, got {price}.")
             return None

        # --- Format Parameters using Exchange Precision ---
        order_size_str = self._format_value(order_size, 'amount')
        price_str = self._format_value(price, 'price') if price else None
        sl_price_str = self._format_value(stop_loss_price, 'price') if stop_loss_price else None
        tp_price_str = self._format_value(take_profit_price, 'price') if take_profit_price else None

        # Check if formatting failed
        if order_size_str is None:
             logger.error("Failed to format order size according to precision.")
             return None
        if order_type == OrderType.LIMIT and price_str is None:
             logger.error("Failed to format limit order price according to precision.")
             return None
        if stop_loss_price and sl_price_str is None:
             logger.warning("Failed to format stop loss price, SL will not be attached.")
             # Decide whether to proceed without SL or fail? Proceeding for now.
        if take_profit_price and tp_price_str is None:
             logger.warning("Failed to format take profit price, TP will not be attached.")
             # Proceeding without TP for now.

        # Convert formatted strings back to float where needed by CCXT methods
        order_size_float = float(order_size_str)
        price_float = float(price_str) if price_str else None

        # --- Prepare CCXT `params` for SL/TP ---
        # This part is highly exchange-specific. Using a common structure first.
        params = {}
        stop_loss_params = {}
        take_profit_params = {}

        if sl_price_str:
            stop_loss_params = {
                'triggerPrice': sl_price_str,
                'price': sl_price_str, # For stop-market orders, price might be ignored or used for limit stops
                'type': 'stop_market', # Or 'stop_limit', depending on desired behavior and exchange support
            }
            # Common CCXT unified parameter (may not be supported by all)
            params['stopLoss'] = stop_loss_params
            # Exchange-specific examples (add more as needed)
            if self.exchange.id == 'bybit':
                params['stopLoss'] = sl_price_str # Bybit unified often uses direct price string
                # params['slTriggerBy'] = 'LastPrice' # Example: specify trigger type if needed
            elif self.exchange.id == 'binance':
                 # Binance often uses 'stopPrice' for trigger and 'price' for limit price if stop-limit
                 params['stopPrice'] = sl_price_str
                 params['type'] = 'STOP_MARKET' # Or 'TAKE_PROFIT_MARKET'

        if tp_price_str:
            take_profit_params = {
                'triggerPrice': tp_price_str,
                'price': tp_price_str,
                'type': 'take_profit_market', # Or 'take_profit_limit'
            }
            params['takeProfit'] = take_profit_params
            if self.exchange.id == 'bybit':
                params['takeProfit'] = tp_price_str
                # params['tpTriggerBy'] = 'LastPrice'
            elif self.exchange.id == 'binance':
                 params['stopPrice'] = tp_price_str # Binance uses stopPrice for TP triggers too
                 params['type'] = 'TAKE_PROFIT_MARKET'

        # Clean up params if SL/TP formatting failed
        if stop_loss_price and sl_price_str is None: params.pop('stopLoss', None)
        if take_profit_price and tp_price_str is None: params.pop('takeProfit', None)
        if self.exchange.id == 'binance' and ('stopPrice' in params and (not sl_price_str and not tp_price_str)):
             params.pop('stopPrice', None) # Clean up binance specific if trigger price missing
             params.pop('type', None)

        # Log the final parameters being sent (be careful with sensitive info if any)
        logger.debug(f"Order params prepared: {params}")


        # --- Simulation Mode ---
        if self.simulation_mode:
            # Determine simulated fill price
            simulated_fill_price = None
            if order_type == OrderType.LIMIT:
                # Simulate fill only if market price crosses the limit price (simplistic)
                current_market_price = self.fetch_market_price() # Fetch current price for simulation logic
                if current_market_price is None:
                     logger.error("[SIMULATION] Could not fetch market price to evaluate limit order fill.")
                     return None
                if side == OrderSide.BUY and current_market_price <= price_float:
                    simulated_fill_price = price_float
                elif side == OrderSide.SELL and current_market_price >= price_float:
                    simulated_fill_price = price_float
                else:
                    logger.info(f"[SIMULATION] Limit order ({side.value} @ {price_float}) not filled at current market price ({current_market_price}).")
                    # Return a structure indicating an open (unfilled) order? Or None?
                    # For simplicity, returning None, assuming we only care about filled orders in this sim.
                    return None # Or return a simulated 'open' order dict
            else: # Market order
                simulated_fill_price = self.fetch_market_price()
                if simulated_fill_price is None:
                    logger.error("[SIMULATION] Could not fetch market price for simulated market order fill.")
                    return None

            simulated_fill_price = float(simulated_fill_price) # Ensure float
            cost = order_size_float * simulated_fill_price
            fee = cost * self.simulated_fee_rate
            simulated_cost_with_fee = cost + fee if side == OrderSide.BUY else cost - fee

            trade_details = {
                "id": f"sim_{int(time.time() * 1000)}_{side.value[:1]}",
                "symbol": self.symbol,
                "status": "closed", # Simulate immediate fill for simplicity
                "side": side.value,
                "type": order_type.value,
                "amount": order_size_float,
                "price": price_float if order_type == OrderType.LIMIT else simulated_fill_price, # Requested price vs fill price
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
                    "orderType": order_type.value.capitalize(),
                    "execType": "Trade", # Simulate execution type
                    "is_simulated": True,
                },
            }
            log_price = price_str if order_type == OrderType.LIMIT else f"Market (~{simulated_fill_price:.{self.market.get('precision', {}).get('price', 2)}f})"
            logger.info(
                f"{Fore.YELLOW}[SIMULATION] Order Filled: ID: {trade_details['id']}, Type: {trade_details['type']}, "
                f"Side: {trade_details['side'].upper()}, Size: {trade_details['amount']:.{self.market.get('precision', {}).get('amount', 8)}f} {self.market['base']}, "
                f"Fill Price: {log_price}, SL: {sl_price_str or 'N/A'}, TP: {tp_price_str or 'N/A'}, "
                f"Simulated Cost: {trade_details['cost']:.2f} {self.market['quote']} (Fee: {fee:.4f})"
            )
            return trade_details

        # --- Live Trading Mode ---
        else:
            order = None
            order_action = f"{side.value.upper()} {order_type.value.upper()}"
            order_details_log = (
                f"{order_size_str} {self.market['base']}"
                f"{f' @ {price_str}' if order_type == OrderType.LIMIT else ''}"
                f"{f' SL:{sl_price_str}' if sl_price_str else ''}"
                f"{f' TP:{tp_price_str}' if tp_price_str else ''}"
                f" (Params: {params})"
            )

            try:
                logger.info(f"{Fore.CYAN}Placing LIVE {order_action} order for {order_details_log}")

                if order_type == OrderType.MARKET:
                    # Use create_order for more flexibility with params across exchanges
                    order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side=side.value,
                        amount=order_size_float,
                        params=params
                    )
                elif order_type == OrderType.LIMIT:
                    order = self.exchange.create_order(
                        symbol=self.symbol,
                        type='limit',
                        side=side.value,
                        amount=order_size_float,
                        price=price_float,
                        params=params
                    )

                if order:
                    # Log details from the returned order structure
                    order_id = order.get('id', 'N/A')
                    order_status = order.get('status', 'N/A')
                    avg_price = order.get('average')
                    filled_amount = order.get('filled', 0.0)
                    cost = order.get('cost') # Cost might include fees depending on exchange

                    log_price_actual = avg_price or order.get('price') # Use average if available, else requested price
                    log_price_display = f"{log_price_actual:.{self.market.get('precision', {}).get('price', 2)}f}" if log_price_actual else (price_str if order_type == OrderType.LIMIT else 'Market')

                    logger.info(
                        f"{Fore.GREEN}LIVE Order Response: ID: {order_id}, Type: {order.get('type')}, Side: {order.get('side', '').upper()}, "
                        f"Amount: {order.get('amount'):.{self.market.get('precision', {}).get('amount', 8)}f}, Filled: {filled_amount:.{self.market.get('precision', {}).get('amount', 8)}f} {self.market['base']}, "
                        f"Avg Price: {log_price_display}, Cost: {cost:.2f} {self.market['quote'] if cost else ''}, "
                        # f"SL: {params.get('stopLoss', 'N/A')}, TP: {params.get('takeProfit', 'N/A')}, " # Params might not reflect actual order state
                        f"Status: {order_status}"
                    )
                    # TODO: Add robust tracking of the actual order state by fetching order status later if needed.
                    # self.open_positions.append(order) # Manage this in the main loop or a dedicated manager
                    return order
                else:
                    # This case might occur if the API call succeeded (no exception) but CCXT returned None/empty
                    logger.error(f"LIVE Order placement call for {order_action} returned no result or an empty order structure.")
                    return None

            except ccxt.InsufficientFunds as e:
                 # This specific error might not be caught by the retry decorator if it's configured not to retry it.
                 logger.error(f"{Fore.RED}LIVE Order Failed: Insufficient funds for {order_action} {order_details_log}. Error: {e}")
                 return None
            except ccxt.InvalidOrder as e:
                 logger.error(f"{Fore.RED}LIVE Order Failed: Invalid order parameters for {order_action} {order_details_log}. Check size, price, limits. Error: {e}")
                 return None
            except ccxt.ExchangeError as e:
                 # Catch other exchange errors that might occur during placement
                 logger.error(f"{Fore.RED}LIVE Order Failed: Exchange error during placement of {order_action} {order_details_log}. Error: {e}")
                 return None
            # NetworkError and RateLimitExceeded should be handled by the decorator
            except Exception as e:
                 logger.exception(f"{Fore.RED}LIVE Order Failed: An unexpected error occurred during order placement of {order_action} {order_details_log}.")
                 return None


    # --- Position Management (Placeholder) ---
    def manage_open_positions(self, current_price: float):
         """
         Placeholder for managing open positions.

         This should:
         1. Fetch current open positions/orders (from exchange or internal state).
         2. Check if Stop Loss or Take Profit levels have been hit.
         3. Check for time-based exits.
         4. Update trailing stop loss levels.
         5. Place closing orders if exit conditions are met.
         6. Update internal state (self.open_positions).
         """
         logger.debug("Checking open positions (Placeholder - No logic implemented)...")
         # Example: Iterate through internally tracked positions
         positions_to_close = []
         for position in self.open_positions:
             # --- This requires a well-defined position structure ---
             # Example structure: {'id': '...', 'side': OrderSide.BUY, 'entry_price': ..., 'size': ..., 'sl': ..., 'tp': ..., 'entry_time': ... , 'trailing_sl_activation_price': ..., 'highest_price_since_entry': ...}

             # Check SL/TP (simple example)
             # if position['side'] == OrderSide.BUY:
             #     if position.get('sl') and current_price <= position['sl']:
             #         logger.info(f"Stop Loss triggered for position {position['id']} at price {current_price}")
             #         positions_to_close.append(position)
             #     elif position.get('tp') and current_price >= position['tp']:
             #          logger.info(f"Take Profit triggered for position {position['id']} at price {current_price}")
             #          positions_to_close.append(position)
             # elif position['side'] == OrderSide.SELL:
             #      # Similar checks for sell positions
             #      pass

             # Check Time-based Exit
             # if self.time_based_exit_minutes > 0:
             #      entry_time = position.get('entry_time') # Assuming stored as timestamp
             #      if entry_time and (time.time() - entry_time) > self.time_based_exit_minutes * 60:
             #           logger.info(f"Time-based exit triggered for position {position['id']}")
             #           positions_to_close.append(position)

             # Check/Update Trailing Stop Loss
             # if self.trailing_stop_loss_percentage > 0 and position.get('is_trailing_active'):
             #      # Logic to update SL based on highest price (for buys) or lowest price (for sells)
             #      pass


         # Close positions identified
         for position in positions_to_close:
              logger.info(f"Attempting to close position {position.get('id', 'N/A')}...")
              # Determine close side (opposite of entry)
              close_side = OrderSide.SELL if position.get('side') == OrderSide.BUY else OrderSide.BUY
              # Place market order to close
              close_order_result = self.place_order(
                  side=close_side,
                  order_size=position.get('size', 0), # Ensure size is stored correctly
                  order_type=OrderType.MARKET
              )
              if close_order_result:
                   logger.info(f"Successfully placed closing order for position {position.get('id')}")
                   # Remove from open positions list (or mark as closed)
                   # self.open_positions.remove(position) # Be careful modifying list while iterating or use index
              else:
                   logger.error(f"Failed to place closing order for position {position.get('id')}")

         # TODO: Implement robust fetching of actual positions/orders from the exchange,
         # reconciliation with internal state, and handling of partial fills.


    # --- Main Loop Logic ---

    def run(self):
        """Main execution loop for the bot."""
        logger.info(f"Starting Scalping Bot - {self.symbol} - Loop Delay: {self.trade_loop_delay}s")
        if self.simulation_mode:
             logger.warning(f"{Fore.YELLOW}--- RUNNING IN SIMULATION MODE ---")
             logger.info(f"Simulated Balance: {self.simulated_balance}, Fee Rate: {self.simulated_fee_rate:.3%}")

        while True:
            self.iteration += 1
            start_time = time.monotonic() # Use monotonic clock for measuring intervals
            logger.info(f"\n----- Iteration {self.iteration} | Time: {pd.Timestamp.now(tz='UTC').isoformat()} -----")

            try:
                # 1. Fetch Data
                # Use asyncio gather in future for concurrent fetches if switching to async
                current_price = self.fetch_market_price()
                orderbook_imbalance = self.fetch_order_book()
                # Fetch only the required amount of data each time
                historical_data = self.fetch_historical_data()

                if current_price is None or historical_data is None:
                    logger.warning("Could not fetch required market data (price or history). Skipping iteration.")
                    self._wait_for_next_iteration(start_time)
                    continue

                # Optional: Check if new candle has arrived to avoid redundant calculations
                current_candle_ts_ms = historical_data.index[-1].value // 10**6 # Timestamp in milliseconds
                if self.last_candle_ts is not None and current_candle_ts_ms <= self.last_candle_ts:
                     logger.debug(f"No new candle data since last iteration ({pd.Timestamp(self.last_candle_ts, unit='ms', tz='UTC')}). Waiting...")
                     self._wait_for_next_iteration(start_time)
                     continue
                logger.debug(f"New candle detected: {historical_data.index[-1]}")
                self.last_candle_ts = current_candle_ts_ms


                # 2. Calculate Indicators
                # Pass the fetched dataframe to calculation methods
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
                # Check if any critical indicator failed
                if any(v is None for k, v in indicators.items() if k != 'volatility' and k != 'macd_hist'): # Volatility/Hist might be optional for signal
                     logger.warning("One or more critical indicators failed calculation. Skipping signal generation.")
                     self._wait_for_next_iteration(start_time)
                     continue

                # 3. Manage Open Positions (Check for Exits: SL/TP/Trailing/Time)
                # This needs to be implemented properly based on actual position tracking
                self.manage_open_positions(current_price) # Placeholder call

                # 4. Generate Trade Signal (Entry Logic)
                # Check if max positions limit reached *after* managing exits
                # TODO: Update self.open_positions accurately in manage_open_positions
                current_open_count = len(self.open_positions) # Needs accurate tracking
                if current_open_count >= self.max_open_positions:
                    logger.info(f"Max open positions ({self.max_open_positions}) reached. Holding.")
                else:
                    score, reasons = self.compute_trade_signal_score(current_price, indicators, orderbook_imbalance)

                    # Define entry thresholds (example: require a strong signal, make configurable?)
                    buy_threshold = 2
                    sell_threshold = -2
                    trade_side: Optional[OrderSide] = None

                    if score >= buy_threshold:
                        trade_side = OrderSide.BUY
                        logger.info(f"{Fore.GREEN}Potential BUY signal (Score: {score}). Reasons: {'; '.join(reasons)}")
                    elif score <= sell_threshold:
                        trade_side = OrderSide.SELL
                        logger.info(f"{Fore.RED}Potential SELL signal (Score: {score}). Reasons: {'; '.join(reasons)}")
                    else:
                        logger.info(f"Neutral signal (Score: {score}). Holding.")

                    # 5. Place Order if Signal Strong Enough and Position Limit Not Reached
                    if trade_side:
                        order_size = self.calculate_order_size(current_price, volatility)

                        if order_size is not None and order_size > 0:
                            # Calculate SL/TP prices based on current price and percentages
                            sl_price, tp_price = None, None
                            price_precision_digits, _ = self._get_market_precision()
                            price_format_str = f"{{:.{price_precision_digits}f}}" if price_precision_digits is not None else "{:.8f}" # Default precision if unknown

                            if trade_side == OrderSide.BUY:
                                if self.stop_loss_pct > 0:
                                    sl_price = current_price * (1 - self.stop_loss_pct)
                                if self.take_profit_pct > 0:
                                    tp_price = current_price * (1 + self.take_profit_pct)
                                # Adjust limit order entry price slightly below current for buys
                                entry_price = current_price * (1 - self.limit_order_offset_buy_pct) if self.entry_order_type == OrderType.LIMIT else None
                            else: # SELL
                                if self.stop_loss_pct > 0:
                                    sl_price = current_price * (1 + self.stop_loss_pct)
                                if self.take_profit_pct > 0:
                                    tp_price = current_price * (1 - self.take_profit_pct)
                                # Adjust limit order entry price slightly above current for sells
                                entry_price = current_price * (1 + self.limit_order_offset_sell_pct) if self.entry_order_type == OrderType.LIMIT else None

                            # Format calculated SL/TP/Entry prices before passing to place_order
                            # place_order will handle final formatting with _format_value
                            logger.debug(f"Calculated Entry: {entry_price}, SL: {sl_price}, TP: {tp_price} (before precision)")

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
                                logger.info(f"Successfully processed {trade_side.value.upper()} order placement request.")
                                # TODO: Add order_result to self.open_positions for tracking
                                # Need a robust way to track simulated vs real positions/orders
                                # Consider storing key details: id, side, size, entry_price, status, timestamp
                                # Example:
                                # position_data = {
                                #    'id': order_result.get('id'),
                                #    'side': trade_side,
                                #    'size': order_result.get('amount'),
                                #    'entry_price': order_result.get('average') or order_result.get('price'),
                                #    'status': order_result.get('status', 'unknown'), # Track status
                                #    'entry_time': time.time(),
                                #    'sl': sl_price, # Store intended SL/TP
                                #    'tp': tp_price,
                                #    'is_simulated': self.simulation_mode
                                # }
                                # self.open_positions.append(position_data)
                            else:
                                logger.error(f"Failed to place {trade_side.value.upper()} order.")
                        else:
                             logger.warning("Order size calculation failed or resulted in zero/negative size. No order placed.")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down gracefully...")
                break
            except ccxt.AuthenticationError as e:
                 # Critical error, likely requires user intervention
                 logger.critical(f"CRITICAL: Authentication failed during main loop: {e}. Check API keys/permissions. Exiting.")
                 break # Exit loop on critical auth error
            except ccxt.ExchangeError as e:
                 # Log exchange errors but potentially continue if recoverable
                 logger.error(f"Exchange error occurred in main loop: {e}. Continuing...")
                 # Consider adding logic to halt trading temporarily if errors persist
            except ccxt.NetworkError as e:
                 logger.error(f"Network error occurred in main loop: {e}. Continuing...")
            except Exception as e:
                # Catch-all for unexpected errors
                logger.exception("An unexpected error occurred in the main loop.")
                # Decide whether to continue or break on general errors
                # Consider adding a maximum consecutive error count before exiting
                # break # Option to stop on any error

            # --- Loop Delay ---
            self._wait_for_next_iteration(start_time)

        logger.info("Scalping Bot stopped.")

    def _wait_for_next_iteration(self, loop_start_time: float):
        """Calculates elapsed time and sleeps for the remaining loop delay."""
        end_time = time.monotonic()
        elapsed = end_time - loop_start_time
        sleep_time = max(0, self.trade_loop_delay - elapsed)
        logger.debug(f"Iteration took {elapsed:.3f}s. Sleeping for {sleep_time:.3f}s...")
        if sleep_time > 0:
            time.sleep(sleep_time)


# --- Main Execution ---
if __name__ == "__main__":
    exit_code = 0
    try:
        # Determine config file path (e.g., from command line argument or default)
        config_arg = sys.argv[1] if len(sys.argv) > 1 else CONFIG_FILE_DEFAULT
        config_path = Path(config_arg)

        bot = ScalpingBot(config_file=config_path)
        bot.run()
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: Configuration file not found. {e}")
        exit_code = 1
    except (ConfigValidationError, yaml.YAMLError, ValueError) as e:
        # Catch config-related errors specifically during init
        logger.error(f"Initialization failed due to invalid configuration: {e}")
        exit_code = 1
    except (ccxt.AuthenticationError, ccxt.ExchangeError, ccxt.NetworkError, ccxt.NotSupported, AttributeError) as e:
         # Catch CCXT/exchange related errors during init
         logger.error(f"Initialization failed due to exchange connection or setup issue: {e}")
         exit_code = 1
    except Exception as e:
        # Catch any other unexpected error during initialization or runtime that wasn't handled in the loop
        logger.exception(f"An unexpected critical error occurred: {e}")
        exit_code = 1
    finally:
        logger.info(f"Exiting Scalping Bot with code {exit_code}.")
        logging.shutdown() # Ensure logs are flushed
        sys.exit(exit_code)
