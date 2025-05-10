import logging
import logging.handlers
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Callable

import ccxt
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init

# --- Initialize Colorama ---
# Ensures colored output works on different terminals and resets style after each print
colorama_init(autoreset=True)

# --- Constants ---
CONFIG_FILE_DEFAULT = Path("config.yaml")
LOG_FILE_DEFAULT = Path("scalping_bot.log")
LOG_MAX_BYTES_DEFAULT = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT_DEFAULT = 5
API_MAX_RETRIES_DEFAULT = 3
API_INITIAL_DELAY_SECONDS_DEFAULT = 1.0
MIN_HISTORICAL_DATA_BUFFER_DEFAULT = (
    10  # Minimum extra data points beyond strict requirement for indicators
)
DEFAULT_PRECISION_FALLBACK = (
    8  # Default decimal places if market precision is unavailable
)


# --- Enums ---
# Using Enums improves type safety and readability for fixed sets of values
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


# --- Configuration Keys (Used for structure and validation) ---
# Centralized keys prevent typos and make refactoring easier
EXCHANGE_SECTION = "exchange"
TRADING_SECTION = "trading"
ORDER_BOOK_SECTION = "order_book"
INDICATORS_SECTION = "indicators"
RISK_MANAGEMENT_SECTION = "risk_management"
LOGGING_SECTION = "logging"

# --- Setup Logger ---
# Global logger instance, configured once by setup_logger after config load
logger = logging.getLogger("ScalpingBot")
logger.setLevel(logging.DEBUG)  # Set default level, will be overridden by config

# Prevent adding handlers multiple times if script is reloaded or function called again
if not logger.hasHandlers():
    # Add a NullHandler initially to prevent "No handler found" warnings
    # before setup_logger is called with actual configuration.
    logger.addHandler(logging.NullHandler())


def setup_logger(
    level: int = logging.INFO,
    log_file: Path = LOG_FILE_DEFAULT,
    max_bytes: int = LOG_MAX_BYTES_DEFAULT,
    backup_count: int = LOG_BACKUP_COUNT_DEFAULT,
) -> None:
    """
    Configures the global 'ScalpingBot' logger with console and rotating file handlers.

    Removes existing handlers before adding new ones to allow reconfiguration.

    Args:
        level: The logging level for the file handler (e.g., logging.INFO, logging.DEBUG).
               Console handler level is INFO unless file level is DEBUG.
        log_file: Path to the log file.
        max_bytes: Maximum size of the log file before rotation.
        backup_count: Number of backup log files to keep.
    """
    # Remove existing handlers before configuring new ones
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()  # Ensure handlers release file resources

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler (INFO level by default, unless DEBUG is explicitly set for file)
    console_level = logging.DEBUG if level == logging.DEBUG else logging.INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    # Rotating File Handler
    try:
        log_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure log directory exists
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)  # Log everything to file based on the main level
        logger.addHandler(file_handler)
        logger.info(
            f"File logging configured: Level={logging.getLevelName(level)}, File={log_file}, MaxBytes={max_bytes}, Backups={backup_count}"
        )
    except IOError as e:
        logger.error(f"Failed to create rotating log file handler for {log_file}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error setting up file logger: {e}", exc_info=True)

    # Set the logger's effective level AFTER handlers are configured
    logger.setLevel(level)


# --- API Retry Decorator ---
def retry_api_call(
    max_retries: int = API_MAX_RETRIES_DEFAULT,
    initial_delay: float = API_INITIAL_DELAY_SECONDS_DEFAULT,
    allowed_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.RateLimitExceeded,
        ccxt.NetworkError,  # Includes RequestTimeout, DDoSProtection, etc.
        ccxt.ExchangeNotAvailable,
        # Add other potentially transient ccxt errors if needed
    ),
    logged_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.ExchangeError,  # Log other exchange errors but don't retry by default unless in allowed_exceptions
    ),
):
    """
    Decorator to retry CCXT API calls with exponential backoff.

    Handles specified transient CCXT errors. Logs warnings/errors and returns None
    after max retries or for non-retried exceptions.

    Args:
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds before the first retry.
        allowed_exceptions: Tuple of exception types to retry.
        logged_exceptions: Tuple of exception types to log as errors but not retry by default.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Optional[Any]:
            retries = 0
            delay = initial_delay
            last_exception: Optional[Exception] = None

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"{type(e).__name__} during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                    # Fall through to sleep and retry logic
                except logged_exceptions as e:
                    last_exception = e
                    err_str = str(e).lower()
                    # Handle specific non-retryable or expected ExchangeErrors
                    if (
                        "order not found" in err_str
                        or "order does not exist" in err_str
                    ):
                        logger.warning(
                            f"Order not found/does not exist for {func.__name__} (likely filled/cancelled): {e}. Returning None."
                        )
                        return None  # Not an error state needing retry
                    elif (
                        "insufficient balance" in err_str
                        or "insufficient funds" in err_str
                        or "insufficient margin" in err_str
                    ):
                        logger.error(
                            f"Insufficient funds for {func.__name__}: {e}. Aborting call."
                        )
                        return None  # Indicate failure, do not retry
                    elif (
                        "Invalid order" in str(e)
                        or "Order size" in str(e)
                        or "Price precision" in str(e)
                    ):
                        logger.error(
                            f"Invalid order parameters for {func.__name__}: {e}. Aborting call."
                        )
                        return None  # Indicate failure, do not retry
                    else:
                        # Log other ExchangeErrors but don't retry by default
                        logger.error(
                            f"Exchange error during {func.__name__}: {e}. Not retrying."
                        )
                        return None  # Indicate failure

                except Exception as e:  # Catch any other unexpected exception
                    last_exception = e
                    logger.exception(  # Use exception() to include traceback
                        f"Unexpected error during {func.__name__}: {e}. Retrying in {delay:.1f}s... "
                        f"(Attempt {retries + 1}/{max_retries + 1})"
                    )
                    # Treat unexpected errors as potentially transient and retry

                # --- Retry Logic ---
                if retries < max_retries:
                    time.sleep(delay)
                    # Exponential backoff with jitter and cap
                    delay = min(delay * 2 + (np.random.rand() * 0.5), 30)
                    retries += 1
                else:
                    break  # Exit loop if max retries reached

            logger.error(
                f"Max retries ({max_retries}) reached for API call {func.__name__}. "
                f"Last error: {type(last_exception).__name__}: {last_exception}. Aborting call."
            )
            return None  # Indicate failure after all retries

        return wrapper

    return decorator


# --- Configuration Validation Schema ---
# Defines the expected structure, types, and constraints for config.yaml.
# Makes validation more declarative and easier to maintain.
CONFIG_SCHEMA = {
    EXCHANGE_SECTION: {
        "required": True,
        "params": {
            "exchange_id": {"type": str, "required": True, "non_empty": True},
            "api_key_env": {
                "type": str,
                "required": False,
                "default": "EXCHANGE_API_KEY",
            },
            "api_secret_env": {
                "type": str,
                "required": False,
                "default": "EXCHANGE_API_SECRET",
            },
            "ccxt_options": {
                "type": dict,
                "required": False,
                "default": {"enableRateLimit": True},
            },
        },
    },
    TRADING_SECTION: {
        "required": True,
        "params": {
            "symbol": {"type": str, "required": True, "non_empty": True},
            "simulation_mode": {"type": bool, "required": True},
            "entry_order_type": {
                "type": str,
                "required": True,
                "allowed": [e.value for e in OrderType],
            },
            "limit_order_offset_buy_pct": {
                "type": (int, float),
                "required": True,
                "range": (0.0, 1.0),
            },
            "limit_order_offset_sell_pct": {
                "type": (int, float),
                "required": True,
                "range": (0.0, 1.0),
            },
            "trade_loop_delay_seconds": {
                "type": (int, float),
                "required": False,
                "default": 10.0,
                "range": (0.1, float("inf")),
            },
        },
    },
    ORDER_BOOK_SECTION: {
        "required": True,
        "params": {
            "depth": {"type": int, "required": True, "range": (1, 1000)},
            "imbalance_threshold": {
                "type": (int, float),
                "required": True,
                "range": (0.0, float("inf")),
            },  # Ratio > threshold = buy signal
        },
    },
    INDICATORS_SECTION: {
        "required": True,
        "params": {
            "timeframe": {
                "type": str,
                "required": True,
                "non_empty": True,
            },  # e.g., '1m', '5m', '1h'
            "volatility_window": {"type": int, "required": True, "range": (2, 1000)},
            "volatility_multiplier": {
                "type": (int, float),
                "required": True,
                "range": (0.0, float("inf")),
            },
            "ema_period": {"type": int, "required": True, "range": (1, 1000)},
            "rsi_period": {"type": int, "required": True, "range": (2, 1000)},
            "macd_short_period": {"type": int, "required": True, "range": (1, 1000)},
            "macd_long_period": {"type": int, "required": True, "range": (1, 1000)},
            "macd_signal_period": {"type": int, "required": True, "range": (1, 1000)},
            "stoch_rsi_period": {"type": int, "required": True, "range": (2, 1000)},
            "stoch_rsi_smooth_k": {
                "type": int,
                "required": False,
                "default": 3,
                "range": (1, 100),
            },
            "stoch_rsi_smooth_d": {
                "type": int,
                "required": False,
                "default": 3,
                "range": (1, 100),
            },
        },
        "custom_validations": [
            lambda cfg: cfg["macd_short_period"] < cfg["macd_long_period"]
            or "MACD short period must be less than long period"
        ],
    },
    RISK_MANAGEMENT_SECTION: {
        "required": True,
        "params": {
            "order_size_percentage": {
                "type": (int, float),
                "required": True,
                "range": (0.0001, 1.0),
            },  # Min > 0
            "stop_loss_percentage": {
                "type": (int, float),
                "required": True,
                "range": (0.0, 1.0),
            },  # 0 means disabled
            "take_profit_percentage": {
                "type": (int, float),
                "required": True,
                "range": (0.0, 1.0),
            },  # 0 means disabled
            "max_open_positions": {"type": int, "required": True, "range": (1, 100)},
            "time_based_exit_minutes": {
                "type": int,
                "required": False,
                "default": 0,
                "range": (0, 10080),
            },  # 0 means disabled, max 1 week
            "trailing_stop_loss_percentage": {
                "type": (int, float),
                "required": False,
                "default": 0.0,
                "range": (0.0, 1.0),
            },  # 0 means disabled
            "simulated_balance": {
                "type": (int, float),
                "required": False,
                "default": 10000.0,
                "range": (0.0, float("inf")),
            },
            "simulated_fee_rate": {
                "type": (int, float),
                "required": False,
                "default": 0.001,
                "range": (0.0, 0.1),
            },  # Realistic fee range
        },
    },
    LOGGING_SECTION: {
        "required": False,  # Optional section
        "params": {
            "level": {
                "type": str,
                "required": False,
                "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO",
            },
            "log_file_path": {
                "type": str,
                "required": False,
                "default": str(LOG_FILE_DEFAULT),
            },
            "log_max_bytes": {
                "type": int,
                "required": False,
                "default": LOG_MAX_BYTES_DEFAULT,
                "range": (1024, float("inf")),
            },
            "log_backup_count": {
                "type": int,
                "required": False,
                "default": LOG_BACKUP_COUNT_DEFAULT,
                "range": (0, 100),
            },
        },
    },
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
            ccxt.AuthenticationError: If API keys are invalid (during exchange init).
            ccxt.ExchangeError: If there's an issue connecting to the exchange (during init).
            ValueError: If the trading symbol is invalid for the exchange.
        """
        self.config_path = Path(config_file)
        self.config: Dict[str, Any] = (
            self._load_and_validate_config()
        )  # Raises errors on failure

        # --- Configure Logger based on validated config ---
        self._configure_logging()

        # --- Assign validated config values to attributes ---
        self._assign_config_attributes()

        # --- Initialize Exchange ---
        self.exchange: ccxt.Exchange = (
            self._initialize_exchange()
        )  # Raises exceptions on failure
        self.market: Dict[str, Any] = (
            self._load_market_details()
        )  # Raises ValueError if symbol invalid

        # --- State Variables ---
        # TODO: Implement robust position tracking, potentially in a separate class
        self.open_positions: List[Dict[str, Any]] = []  # Stores active positions/orders
        self.iteration: int = 0
        self.daily_pnl: float = 0.0  # Placeholder, needs calculation logic
        self.last_candle_ts: Optional[int] = (
            None  # Track last processed candle timestamp (ms)
        )
        self.simulated_balance_current: float = (
            self.simulated_balance
        )  # Track current sim balance

        logger.info(
            f"ScalpingBot initialized for {self.symbol} on {self.exchange_id}. Simulation Mode: {self.simulation_mode}"
        )

    def _configure_logging(self) -> None:
        """Configures the global logger using settings from the validated config."""
        log_cfg = self.config.get(
            LOGGING_SECTION, {}
        )  # Use validated defaults if section missing
        log_level_name = log_cfg.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        log_file = Path(log_cfg.get("log_file_path", LOG_FILE_DEFAULT))
        log_max_bytes = log_cfg.get("log_max_bytes", LOG_MAX_BYTES_DEFAULT)
        log_backup_count = log_cfg.get("log_backup_count", LOG_BACKUP_COUNT_DEFAULT)

        setup_logger(
            level=log_level,
            log_file=log_file,
            max_bytes=log_max_bytes,
            backup_count=log_backup_count,
        )
        # Logger level already set in setup_logger

    def _assign_config_attributes(self) -> None:
        """Assigns validated configuration values to instance attributes for easier access."""
        # Using .get() with defaults derived from the schema validation process ensures attributes exist
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
        self.limit_order_offset_buy_pct: float = float(
            trade_cfg["limit_order_offset_buy_pct"]
        )
        self.limit_order_offset_sell_pct: float = float(
            trade_cfg["limit_order_offset_sell_pct"]
        )
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
        self.trailing_stop_loss_percentage: float = float(
            risk_cfg["trailing_stop_loss_percentage"]
        )
        self.simulated_balance: float = float(risk_cfg["simulated_balance"])
        self.simulated_fee_rate: float = float(risk_cfg["simulated_fee_rate"])

    def _validate_config_section(
        self, section_name: str, section_config: Any, section_schema: Dict
    ) -> Dict[str, Any]:
        """Validates a single section of the configuration against its schema."""
        if not isinstance(section_config, dict):
            raise ConfigValidationError(
                f"Section '{section_name}' must be a dictionary."
            )

        validated_section = {}
        param_schema = section_schema.get("params", {})

        # Validate parameters within the section
        for param, schema in param_schema.items():
            value = section_config.get(param)

            # Check required parameters and apply defaults
            if value is None:
                if schema.get("required", False):
                    raise ConfigValidationError(
                        f"Missing required parameter '{param}' in section '{section_name}'."
                    )
                elif "default" in schema:
                    validated_section[param] = schema["default"]
                    logger.debug(
                        f"Using default value for '{section_name}.{param}': {schema['default']}"
                    )
                else:
                    validated_section[param] = (
                        None  # Optional parameter without default
                    )
                continue  # Skip further checks for this parameter

            # Type check
            expected_type = schema["type"]
            if not isinstance(value, expected_type):
                # Allow int to be validated as float if float is expected type
                if (
                    isinstance(expected_type, tuple)
                    and float in expected_type
                    and isinstance(value, int)
                ):
                    value = float(value)  # Coerce int to float for validation
                elif not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Parameter '{param}' in '{section_name}' must be type {expected_type}, got {type(value)}."
                    )

            # Non-empty check for strings
            if (
                schema.get("non_empty", False)
                and isinstance(value, str)
                and not value.strip()
            ):
                raise ConfigValidationError(
                    f"Parameter '{param}' in '{section_name}' cannot be empty."
                )

            # Allowed values check
            allowed = schema.get("allowed")
            if allowed and value not in allowed:
                raise ConfigValidationError(
                    f"Parameter '{param}' in '{section_name}' must be one of {allowed}, got '{value}'."
                )

            # Range check (for numeric types)
            value_range = schema.get("range")
            if value_range and isinstance(value, (int, float)):
                # Check if value is within the specified range [min_val, max_val]
                min_val, max_val = value_range
                if not (min_val <= value <= max_val):
                    raise ConfigValidationError(
                        f"Parameter '{param}' in '{section_name}' ({value}) must be between {min_val} and {max_val}."
                    )
            elif value_range and not isinstance(value, (int, float)):
                logger.warning(
                    f"Range check skipped for non-numeric parameter '{param}' in '{section_name}'."
                )

            validated_section[param] = (
                value  # Assign validated (and potentially type-coerced) value
            )

        # Check for unexpected parameters in the user's config section
        extra_params = set(section_config.keys()) - set(param_schema.keys())
        if extra_params:
            logger.warning(
                f"Ignoring unexpected parameters in section '{section_name}': {', '.join(extra_params)}"
            )

        # Perform custom validations defined in the schema for this section
        custom_validations = section_schema.get("custom_validations", [])
        for validation_func in custom_validations:
            try:
                result = validation_func(
                    validated_section
                )  # Pass the validated data so far
                if isinstance(result, str):  # Expect error message string on failure
                    raise ConfigValidationError(
                        f"Custom validation failed for section '{section_name}': {result}"
                    )
            except KeyError as e:
                # This occurs if the custom validation function relies on a parameter that wasn't provided (and wasn't required/defaulted)
                raise ConfigValidationError(
                    f"Custom validation for '{section_name}' failed: Missing parameter {e} needed for validation."
                )
            except Exception as e:
                # Catch any other error within the custom validation function itself
                raise ConfigValidationError(
                    f"Error during custom validation for '{section_name}': {e}"
                )

        return validated_section

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Loads the YAML config file and validates it against CONFIG_SCHEMA."""
        logger.info(f"Loading configuration from {self.config_path}...")
        if not self.config_path.is_file():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_raw = yaml.safe_load(f)
            if not isinstance(config_raw, dict):
                raise ConfigValidationError(
                    "Config file content must be a YAML dictionary (key-value pairs)."
                )
            logger.info("Configuration file loaded successfully.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file: {e}")
            raise  # Re-raise the original error for clarity
        except Exception as e:
            logger.error(f"Error reading configuration file {self.config_path}: {e}")
            raise

        logger.info("Validating configuration...")
        validated_config = {}
        all_sections_in_file = set(config_raw.keys())
        defined_sections_in_schema = set(CONFIG_SCHEMA.keys())

        # Iterate through the schema to validate each defined section
        for section_name, section_schema in CONFIG_SCHEMA.items():
            user_section_data = config_raw.get(section_name)

            if user_section_data is None:  # Section not present in user config file
                if section_schema.get("required", False):
                    raise ConfigValidationError(
                        f"Missing required section '{section_name}' in config file."
                    )
                else:
                    # Section is optional, create it with default values from its parameter schema
                    logger.debug(
                        f"Optional section '{section_name}' not found in config file. Applying defaults."
                    )
                    # Pass an empty dict to validation to populate defaults
                    validated_config[section_name] = self._validate_config_section(
                        section_name, {}, section_schema
                    )
            else:
                # Section exists, validate it against its schema
                validated_config[section_name] = self._validate_config_section(
                    section_name, user_section_data, section_schema
                )

        # Check for unexpected top-level sections in the user's config file
        extra_sections = all_sections_in_file - defined_sections_in_schema
        if extra_sections:
            logger.warning(
                f"Ignoring unexpected top-level sections in config file: {', '.join(extra_sections)}"
            )

        logger.info("Configuration validated successfully.")
        return validated_config

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and returns the CCXT exchange instance based on config."""
        logger.info(f"Initializing exchange: {self.exchange_id.upper()}...")
        try:
            exchange_class: Type[ccxt.Exchange] = getattr(ccxt, self.exchange_id)
        except AttributeError:
            msg = f"Exchange ID '{self.exchange_id}' is not a valid CCXT exchange."
            logger.error(msg)
            raise AttributeError(msg)

        api_key = os.getenv(self.api_key_env)
        api_secret = os.getenv(self.api_secret_env)

        # Start with CCXT options from config (e.g., rate limiting)
        exchange_config = self.ccxt_options.copy()

        if not self.simulation_mode:
            if api_key and api_secret:
                exchange_config["apiKey"] = api_key
                exchange_config["secret"] = api_secret
                logger.info(
                    f"LIVE mode: API credentials loaded from environment variables ({self.api_key_env}, {self.api_secret_env})."
                )
            else:
                # Critical warning if running live without keys
                logger.critical(
                    f"{Fore.RED}CRITICAL: Running in LIVE mode but API Key ({self.api_key_env}) or Secret ({self.api_secret_env}) "
                    "environment variables not set or empty. Trading operations WILL fail."
                )
                # Optionally raise an error here to prevent starting in a non-functional live state
                # raise ConfigValidationError("API credentials missing for live mode.")
        else:
            logger.info("SIMULATION mode: No API credentials required or used.")
            # Some exchanges might need dummy keys even for sandbox/testnet
            exchange_config.pop("apiKey", None)
            exchange_config.pop("secret", None)

        try:
            exchange = exchange_class(exchange_config)

            # Enable sandbox/testnet mode if available and in simulation
            if self.simulation_mode:
                # Prefer 'test' environment setting if available (more common than 'sandbox')
                if "test" in exchange.urls:
                    exchange.urls["api"] = exchange.urls["test"]
                    logger.info("Enabled exchange TESTNET mode for simulation.")
                elif exchange.has.get("sandbox"):
                    try:
                        exchange.set_sandbox_mode(True)
                        logger.info("Enabled exchange SANDBOX mode for simulation.")
                    except Exception as sandbox_err:
                        logger.warning(
                            f"Failed to enable sandbox mode (may not be fully supported): {sandbox_err}"
                        )

            logger.debug("Loading markets...")
            exchange.load_markets()  # Fetch available trading pairs, precision, limits etc.
            logger.info(
                f"Connected to {self.exchange_id.upper()} and markets loaded successfully."
            )

            # --- Check Required Exchange Capabilities ---
            # Define capabilities needed for the bot's core functions
            required_capabilities = {
                "fetchOHLCV": True,
                "fetchOrderBook": True,
                "fetchTicker": True,  # Used for current price
                "createOrder": not self.simulation_mode,  # Only required in live mode
                "fetchBalance": not self.simulation_mode,  # Only required in live mode
            }
            missing_critical = []
            for cap, required in required_capabilities.items():
                if required and not exchange.has.get(cap):
                    log_func = logger.error if required else logger.warning
                    msg = (
                        f"Exchange {self.exchange_id} lacks required capability: {cap}"
                    )
                    log_func(msg)
                    if required:
                        missing_critical.append(cap)

            if missing_critical:
                raise ccxt.NotSupported(
                    f"Exchange {self.exchange_id} missing critical capabilities: {', '.join(missing_critical)}"
                )

            # Log optional but useful capabilities
            for cap in [
                "fetchMyTrades",
                "fetchOpenOrders",
                "fetchOrder",
                "cancelOrder",
            ]:
                if not exchange.has.get(cap):
                    logger.warning(
                        f"Exchange {self.exchange_id} does not support '{cap}', position management might be limited."
                    )

            return exchange

        except ccxt.AuthenticationError as e:
            logger.error(
                f"Authentication Error with {self.exchange_id}: {e}. Check API keys/permissions."
            )
            raise  # Propagate specific error
        except ccxt.ExchangeNotAvailable as e:
            logger.error(
                f"Exchange Not Available: {self.exchange_id} might be down or unreachable. {e}"
            )
            raise
        except ccxt.NetworkError as e:
            logger.error(f"Network error during exchange initialization: {e}")
            raise
        except ccxt.ExchangeError as e:  # Catch other specific CCXT exchange errors
            logger.error(f"Exchange error during initialization: {e}")
            raise
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during exchange initialization: {e}"
            )
            raise  # Re-raise unexpected errors

    def _load_market_details(self) -> Dict[str, Any]:
        """Loads and validates market details for the configured symbol from the exchange."""
        logger.debug(f"Loading market details for symbol: {self.symbol}")
        try:
            # Ensure markets are loaded before accessing market details
            if not self.exchange.markets:
                logger.info("Markets not loaded yet, loading now...")
                self.exchange.load_markets()

            market = self.exchange.market(self.symbol)  # Throws BadSymbol if not found
            if market is None:
                # Should be caught by BadSymbol, but check just in case
                raise ValueError(
                    f"Market details not found for symbol '{self.symbol}' even though BadSymbol was not raised."
                )

            # Log key market details for verification
            logger.info(
                f"Market details for {self.symbol}: Base={market.get('base')}, Quote={market.get('quote')}, Active={market.get('active')}"
            )
            logger.debug(
                f"Precision: Amount={market.get('precision', {}).get('amount')}, Price={market.get('precision', {}).get('price')}"
            )
            logger.debug(
                f"Limits: Amount Min/Max={market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Cost Min/Max={market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}"
            )

            # --- Validate Critical Market Info ---
            if not market.get("active", True):  # Assume active if key missing
                logger.warning(
                    f"Market {self.symbol} is marked as inactive by the exchange."
                )
                # Consider raising an error or exiting if trading inactive markets is not desired
                # raise ValueError(f"Market {self.symbol} is inactive.")
            if not market.get("base") or not market.get("quote"):
                raise ValueError(
                    f"Base or Quote currency missing in market details for {self.symbol}"
                )
            if market.get("precision") is None or market.get("limits") is None:
                logger.warning(
                    f"Precision or Limits information might be incomplete for {self.symbol}. Order placement might fail."
                )
            if (
                market.get("precision", {}).get("amount") is None
                or market.get("precision", {}).get("price") is None
            ):
                logger.warning(
                    f"Amount or Price precision is missing for {self.symbol}. Using fallback precision {DEFAULT_PRECISION_FALLBACK}."
                )

            return market

        except (
            ccxt.BadSymbol
        ) as e:  # Catch specific error from market() if symbol not found
            logger.error(
                f"Symbol '{self.symbol}' not found or invalid on {self.exchange_id}."
            )
            available_symbols = list(self.exchange.markets.keys())
            logger.info(
                f"Available symbols sample ({len(available_symbols)} total): {available_symbols[:20]}..."
            )
            # Wrap the original exception for better context
            raise ValueError(
                f"Symbol '{self.symbol}' not available on exchange '{self.exchange_id}'"
            ) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(
                f"Failed to load market details for {self.symbol} due to exchange/network error: {e}"
            )
            raise  # Propagate error, initialization should fail
        except Exception as e:
            logger.exception(
                f"Unexpected error loading market details for {self.symbol}: {e}"
            )
            raise

    # --- Data Fetching Methods ---

    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the last traded price for the symbol using fetch_ticker."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        if not self.exchange.has["fetchTicker"]:
            logger.error(
                "Exchange does not support fetchTicker. Cannot fetch current price."
            )
            return None

        ticker = self.exchange.fetch_ticker(self.symbol)
        if not ticker or not isinstance(ticker, dict):
            logger.warning(
                f"fetch_ticker returned invalid data for {self.symbol}: {ticker}"
            )
            return None

        # Prioritize 'last', then 'close'. Fallback to bid/ask midpoint.
        price = ticker.get("last") or ticker.get("close")
        price_source = "last/close"

        if price is None:
            ask = ticker.get("ask")
            bid = ticker.get("bid")
            if ask is not None and bid is not None and ask > 0 and bid > 0:
                price = (ask + bid) / 2.0
                price_source = "midpoint"
                logger.debug(
                    f"Using midpoint price: {price:.{self._get_precision_digits('price')}f}"
                )
            else:
                logger.warning(
                    f"Market price ('last', 'close', 'ask'/'bid') unavailable in ticker for {self.symbol}. Ticker: {ticker}"
                )
                return None

        try:
            price_float = float(price)
            if price_float <= 0:
                logger.warning(
                    f"Fetched market price ({price_source}) is zero or negative: {price_float}"
                )
                return None
            logger.debug(
                f"Fetched market price ({price_source}): {price_float:.{self._get_precision_digits('price')}f}"
            )
            return price_float
        except (ValueError, TypeError) as e:
            logger.error(
                f"Failed to convert fetched price '{price}' ({price_source}) to float: {e}"
            )
            return None

    @retry_api_call()
    def fetch_order_book(self) -> Optional[float]:
        """Fetches order book and calculates the bid/ask volume imbalance ratio."""
        logger.debug(
            f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})..."
        )
        if not self.exchange.has["fetchOrderBook"]:
            logger.error(
                "Exchange does not support fetchOrderBook. Cannot calculate imbalance."
            )
            return None

        orderbook = self.exchange.fetch_order_book(
            self.symbol, limit=self.order_book_depth
        )

        if not orderbook or not isinstance(orderbook, dict):
            logger.warning(
                f"fetch_order_book returned invalid data for {self.symbol}: {orderbook}"
            )
            return None

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            # Allow calculation even if one side is empty (results in 0 or inf imbalance)
            logger.debug(
                f"Order book data has empty bids or asks for {self.symbol}. Bids: {len(bids)}, Asks: {len(asks)}"
            )
            # Continue to calculation, handle zero volume there

        # Calculate total volume within the specified depth
        # Order book format: [[price, amount], [price, amount], ...]
        try:
            # Ensure amounts are valid numbers before summing
            bid_volume = sum(
                float(bid[1])
                for bid in bids
                if len(bid) > 1 and isinstance(bid[1], (int, float))
            )
            ask_volume = sum(
                float(ask[1])
                for ask in asks
                if len(ask) > 1 and isinstance(ask[1], (int, float))
            )
        except (TypeError, IndexError, ValueError) as e:
            logger.error(
                f"Error processing order book data for volume calculation: {e}. Bids: {bids}, Asks: {asks}"
            )
            return None

        # Calculate imbalance ratio, handle division by zero
        if ask_volume <= 1e-12:  # Use a small threshold to avoid floating point issues
            # Infinite imbalance if bids exist and asks are zero
            # Neutral (1.0) if both bids and asks are zero
            imbalance_ratio = float("inf") if bid_volume > 1e-12 else 1.0
            logger.debug(
                f"Ask volume is effectively zero ({ask_volume}). Setting imbalance ratio to {imbalance_ratio}."
            )
        else:
            imbalance_ratio = bid_volume / ask_volume

        logger.debug(
            f"Order Book - Top {self.order_book_depth} levels: "
            f"Bid Vol: {bid_volume:.{self._get_precision_digits('amount')}f}, Ask Vol: {ask_volume:.{self._get_precision_digits('amount')}f}, "
            f"Bid/Ask Ratio: {imbalance_ratio:.2f}"
        )
        return imbalance_ratio

    @retry_api_call()
    def fetch_historical_data(
        self, limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data for the configured symbol and timeframe.

        Args:
            limit: The number of candles to fetch. If None, calculates based on
                   indicator requirements plus a buffer.

        Returns:
            A pandas DataFrame with OHLCV data indexed by timestamp (UTC),
            cleaned of NaNs and duplicates, or None on failure.
        """
        if not self.exchange.has["fetchOHLCV"]:
            logger.error(
                f"Exchange {self.exchange_id} does not support fetchOHLCV. Cannot fetch historical data."
            )
            return None

        # Determine the minimum number of candles required for all indicators
        # Add buffer for calculations involving differences, shifts, and initial smoothing periods
        min_required = max(
            self.volatility_window + 1,
            self.ema_period + 1,  # EMA needs warmup
            self.rsi_period + 1,  # RSI needs diff
            self.macd_long_period
            + self.macd_signal_period,  # MACD needs long EMA + signal EMA warmup
            self.rsi_period
            + self.stoch_rsi_period
            + self.stoch_rsi_smooth_k
            + self.stoch_rsi_smooth_d,  # Stoch RSI chain needs longest chain
            MIN_HISTORICAL_DATA_BUFFER_DEFAULT,  # Ensure a baseline minimum
        )

        # Fetch slightly more than strictly needed for stability and lookback calculations
        fetch_limit = (
            limit
            if limit is not None
            else min_required + MIN_HISTORICAL_DATA_BUFFER_DEFAULT
        )

        logger.debug(
            f"Fetching {fetch_limit} historical {self.timeframe} candles for {self.symbol} (min required: {min_required})..."
        )

        try:
            # Fetch OHLCV data using CCXT: [timestamp (ms), open, high, low, close, volume]
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=fetch_limit
            )

            if (
                not ohlcv or len(ohlcv) < 2
            ):  # Need at least 2 points for most calculations
                logger.warning(
                    f"Insufficient or no historical OHLCV data returned for {self.symbol}@{self.timeframe}. Got {len(ohlcv) if ohlcv else 0} candles."
                )
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # --- Data Cleaning and Preparation ---
            # 1. Convert timestamp to datetime objects (UTC) and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # 2. Convert OHLCV columns to numeric, coercing errors to NaN
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 3. Drop rows with NaN in critical OHLC columns
            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if len(df) < initial_len:
                logger.debug(
                    f"Dropped {initial_len - len(df)} rows with NaN values in OHLC columns."
                )

            # 4. Handle potential NaN in volume (replace with 0 or forward fill?) - Replacing with 0 is safer for volume-based indicators
            if df["volume"].isnull().any():
                nan_volume_count = df["volume"].isnull().sum()
                df["volume"].fillna(0.0, inplace=True)
                logger.debug(
                    f"Filled {nan_volume_count} NaN values in 'volume' column with 0.0."
                )

            # 5. Drop potential duplicate timestamps (keep first entry)
            initial_len = len(df)
            df = df[~df.index.duplicated(keep="first")]
            if len(df) < initial_len:
                logger.warning(
                    f"Dropped {initial_len - len(df)} duplicate timestamp rows from OHLCV data."
                )

            # 6. Sort by timestamp index just in case API returns unsorted data
            df.sort_index(inplace=True)

            # --- Final Check for Sufficient Data Length ---
            if len(df) < min_required:
                logger.error(
                    f"Insufficient historical data after cleaning. Requested ~{fetch_limit}, got {len(df)}, minimum required {min_required}. Cannot proceed with calculations."
                )
                return None

            logger.debug(
                f"Historical data fetched and processed successfully. Shape: {df.shape}, Time Range: {df.index.min()} to {df.index.max()}"
            )
            return df

        # Specific CCXT errors might be caught by retry decorator, but handle here if they propagate
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.error(
                f"Error fetching OHLCV data for {self.symbol}@{self.timeframe}: {e}"
            )
            return None
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred while fetching or processing historical data for {self.symbol}@{self.timeframe}: {e}"
            )
            return None

    # --- Indicator Calculations ---

    def _calculate_indicator(
        self,
        data: Optional[pd.Series],
        required_length: int,
        calc_func: Callable[..., Optional[Union[pd.Series, float, tuple]]],
        name: str,
        **kwargs,
    ) -> Optional[Any]:
        """
        Helper to safely calculate an indicator on a pandas Series.

        Checks for None input, sufficient data length, executes the calculation,
        extracts the last value if a Series is returned, checks for NaN/inf, and logs.

        Args:
            data: The pandas Series to calculate on (e.g., close prices).
            required_length: Minimum number of data points needed for a reliable calculation.
            calc_func: The function performing the indicator calculation.
                       It should accept the Series as the first argument and any kwargs.
                       Should return the result (Series, scalar, tuple) or None on internal error.
            name: The name of the indicator for logging.
            **kwargs: Additional keyword arguments passed to calc_func.

        Returns:
            The calculated indicator value(s) (usually the last point(s)),
            or None if calculation fails or input is insufficient.
        """
        if data is None:
            logger.debug(f"Input data is None for {name} calculation.")
            return None
        if len(data) < required_length:
            logger.warning(
                f"Insufficient data for {name} calculation (need {required_length}, got {len(data)})"
            )
            return None

        try:
            result = calc_func(data, **kwargs)

            # If calculation function itself returned None, propagate it
            if result is None:
                logger.warning(f"{name} calculation function returned None.")
                return None

            # Extract the last value(s) if the result is a Series or DataFrame column
            final_value: Any = None
            if isinstance(result, pd.Series):
                if result.empty:
                    logger.warning(f"{name} calculation resulted in an empty Series.")
                    return None
                final_value = result.iloc[-1]
            elif isinstance(
                result, tuple
            ):  # Handle functions returning multiple values (like MACD, StochRSI)
                # Check each element of the tuple
                processed_tuple = []
                valid = True
                for item in result:
                    if isinstance(item, pd.Series):
                        if item.empty:
                            valid = False
                            break
                        processed_tuple.append(item.iloc[-1])
                    else:
                        processed_tuple.append(item)  # Assume scalar
                if not valid:
                    logger.warning(
                        f"{name} calculation resulted in an empty Series within the tuple."
                    )
                    return None
                final_value = tuple(processed_tuple)
            else:
                # Assume scalar result or handle other types if necessary
                final_value = result

            # Check for NaN or infinite values in the final result(s)
            if isinstance(final_value, tuple):
                if any(
                    pd.isna(v) or (isinstance(v, (float, int)) and np.isinf(v))
                    for v in final_value
                ):
                    logger.warning(
                        f"{name} calculation resulted in NaN or infinite value(s) in tuple: {final_value}"
                    )
                    return None
            elif pd.isna(final_value) or (
                isinstance(final_value, (float, int)) and np.isinf(final_value)
            ):
                logger.warning(
                    f"{name} calculation resulted in NaN or infinite value: {final_value}"
                )
                return None

            # Log the calculated value(s) with appropriate formatting
            log_msg = f"Calculated {name}: "
            if isinstance(final_value, (float, np.floating)):
                log_msg += f"{final_value:.4f}"  # Default formatting for floats
            elif isinstance(final_value, tuple):
                log_msg += ", ".join(
                    [
                        f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                        for v in final_value
                    ]
                )
            else:
                log_msg += str(final_value)  # For integers or other types
            logger.debug(log_msg)

            return final_value

        except Exception as e:
            # Log calculation errors concisely by default, add exc_info=True for detailed debugging if needed
            logger.error(
                f"Error calculating {name}: {type(e).__name__} - {e}", exc_info=False
            )
            return None

    def calculate_volatility(self, data: pd.DataFrame) -> Optional[float]:
        """Calculates the rolling log return standard deviation (volatility)."""
        if data is None or "close" not in data.columns:
            return None
        required = self.volatility_window + 1  # Need one previous point for shift/diff
        return self._calculate_indicator(
            data["close"],
            required,
            lambda s: np.log(s / s.shift(1))
            .rolling(window=self.volatility_window)
            .std(),
            f"Volatility (window {self.volatility_window})",
        )

    def calculate_ema(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """Calculates the Exponential Moving Average (EMA)."""
        if data is None or "close" not in data.columns:
            return None
        # EMA technically only needs 'period' points, but more is better for stability
        required = period + 1
        return self._calculate_indicator(
            data["close"],
            required,
            lambda s, p=period: s.ewm(
                span=p, adjust=False
            ).mean(),  # Pass period via lambda default or kwargs
            f"EMA (period {period})",
        )

    def calculate_rsi(self, data: pd.DataFrame) -> Optional[float]:
        """Calculates the Relative Strength Index (RSI)."""
        if data is None or "close" not in data.columns:
            return None
        required = self.rsi_period + 1  # For diff() and initial smoothing warmup

        def _rsi_calc(series: pd.Series, period: int) -> Optional[pd.Series]:
            if len(series) < period + 1:
                return None  # Ensure enough data for initial calculation
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).fillna(0.0)
            loss = -delta.where(delta < 0, 0.0).fillna(0.0)

            # Use EWMA for smoothing average gain/loss (common method)
            avg_gain = gain.ewm(
                alpha=1 / period, adjust=False, min_periods=period
            ).mean()
            avg_loss = loss.ewm(
                alpha=1 / period, adjust=False, min_periods=period
            ).mean()

            # Handle case where avg_loss is zero
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

        return self._calculate_indicator(
            data["close"],
            required,
            _rsi_calc,
            f"RSI (period {self.rsi_period})",
            period=self.rsi_period,  # Pass period to the calc function via kwargs
        )

    def calculate_macd(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculates the Moving Average Convergence Divergence (MACD).

        Returns:
            A tuple containing (MACD Line, Signal Line, Histogram), or (None, None, None) on failure.
        """
        if data is None or "close" not in data.columns:
            return None, None, None
        # Required length is roughly long_period + signal_period for stable values after EWMA warmup
        required = self.macd_long_period + self.macd_signal_period

        # Use the _calculate_indicator helper, but the calc function returns a tuple of Series
        def _macd_calc(
            series: pd.Series, short: int, long: int, signal: int
        ) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
            if len(series) < long + signal:
                return None  # Ensure enough data
            ema_short = series.ewm(span=short, adjust=False).mean()
            ema_long = series.ewm(span=long, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        result_tuple = self._calculate_indicator(
            data["close"],
            required,
            _macd_calc,
            "MACD",
            short=self.macd_short_period,
            long=self.macd_long_period,
            signal=self.macd_signal_period,
        )

        return result_tuple if result_tuple is not None else (None, None, None)

    def calculate_stoch_rsi(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates the Stochastic RSI (%K and %D).

        Returns:
            A tuple containing (%K, %D), or (None, None) on failure.
        """
        if data is None or "close" not in data.columns:
            return None, None
        # Stoch RSI needs RSI calculation + rolling window + k smoothing + d smoothing
        # Estimate required length generously
        required = (
            self.rsi_period
            + self.stoch_rsi_period
            + self.stoch_rsi_smooth_k
            + self.stoch_rsi_smooth_d
            + 5
        )  # Extra buffer

        def _stoch_rsi_calc(
            series: pd.Series, rsi_p: int, stoch_p: int, k_smooth: int, d_smooth: int
        ) -> Optional[Tuple[pd.Series, pd.Series]]:
            # 1. Calculate RSI first
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).fillna(0.0)
            loss = -delta.where(delta < 0, 0.0).fillna(0.0)
            avg_gain = gain.ewm(alpha=1 / rsi_p, adjust=False, min_periods=rsi_p).mean()
            avg_loss = loss.ewm(alpha=1 / rsi_p, adjust=False, min_periods=rsi_p).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            rsi_series.dropna(inplace=True)  # Drop initial NaNs from RSI calculation

            # Check if enough RSI values remain for Stoch calculations
            if (
                len(rsi_series) < stoch_p + k_smooth + d_smooth - 2
            ):  # Rolling needs N points, mean needs M points
                logger.warning(
                    f"Insufficient RSI values ({len(rsi_series)}) for Stoch RSI rolling/smoothing."
                )
                return None

            # 2. Calculate Stoch RSI %K (raw) based on RSI series
            min_rsi = rsi_series.rolling(window=stoch_p).min()
            max_rsi = rsi_series.rolling(window=stoch_p).max()
            stoch_rsi_k_raw = (
                100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-10)
            )  # Avoid div by zero

            # 3. Smooth %K and %D using rolling mean
            k = stoch_rsi_k_raw.rolling(window=k_smooth).mean()
            d = k.rolling(window=d_smooth).mean()

            return k, d

        result_tuple = self._calculate_indicator(
            data["close"],
            required,
            _stoch_rsi_calc,
            "Stochastic RSI",
            rsi_p=self.rsi_period,
            stoch_p=self.stoch_rsi_period,
            k_smooth=self.stoch_rsi_smooth_k,
            d_smooth=self.stoch_rsi_smooth_d,
        )

        return result_tuple if result_tuple is not None else (None, None)

    # --- Balance and Order Sizing ---

    @retry_api_call()
    def fetch_balance(self) -> Optional[float]:
        """
        Fetches the available balance for the quote currency.
        Returns the simulated balance if in simulation mode.
        """
        if self.simulation_mode:
            # In simulation mode, return the *current* simulated balance
            logger.debug(
                f"Simulation mode: Using current simulated balance of {self.simulated_balance_current:.{self._get_precision_digits('price')}f} {self.market.get('quote', 'QUOTE')}."
            )
            return self.simulated_balance_current

        # Live mode: Fetch from exchange
        quote_currency = self.market.get("quote")
        if not quote_currency:
            logger.error(
                "Could not determine quote currency from market data. Cannot fetch balance."
            )
            return None

        if not self.exchange.has["fetchBalance"]:
            logger.error(
                "Exchange does not support fetchBalance. Cannot fetch live balance."
            )
            return None

        try:
            balance_info = self.exchange.fetch_balance()

            # Use 'free' balance for placing new orders
            # Handle nested structure gracefully
            free_balance = balance_info.get("free", {}).get(quote_currency)

            if free_balance is None:
                logger.warning(
                    f"Could not find 'free' balance for quote currency '{quote_currency}' in balance response. Response keys: {balance_info.keys()}"
                )
                # Fallback: Check 'total' balance? Or assume 0 free? Assuming 0 is safer.
                total_balance = balance_info.get("total", {}).get(quote_currency, 0.0)
                logger.warning(
                    f"Free balance unavailable, total balance is {total_balance}. Returning 0.0 as available balance."
                )
                free_balance = 0.0  # Safer to assume 0 free if not explicitly found

            balance_float = float(free_balance)
            logger.debug(
                f"Fetched free balance for {quote_currency}: {balance_float:.{self._get_precision_digits('price')}f}"
            )
            return balance_float

        except ccxt.AuthenticationError:
            # This shouldn't be retried by the decorator, log and return None
            logger.error(
                "Authentication required to fetch balance. Check API keys/permissions."
            )
            return None
        # Network/Exchange errors handled by decorator, but catch unexpected here
        except Exception as e:
            logger.exception(
                f"Unexpected error retrieving balance for {quote_currency}: {e}"
            )
            return None

    def _get_precision_digits(self, precision_type: str) -> int:
        """
        Safely retrieves the number of decimal places for amount or price precision.
        Uses market info if available, otherwise falls back to a default.

        Args:
            precision_type: 'amount' or 'price'.

        Returns:
            Number of decimal places (int).
        """
        if precision_type not in ["amount", "price"]:
            raise ValueError("precision_type must be 'amount' or 'price'")

        precision_val = self.market.get("precision", {}).get(precision_type)

        if precision_val is None:
            logger.warning(
                f"Precision value for '{precision_type}' not found in market details for {self.symbol}. Using fallback: {DEFAULT_PRECISION_FALLBACK}"
            )
            return DEFAULT_PRECISION_FALLBACK

        # CCXT precision can be integer digits or decimal places (e.g., 0.001 -> 3)
        try:
            if isinstance(precision_val, int):
                # If it's an integer, assume it represents decimal places (common case)
                return precision_val
            elif isinstance(precision_val, (float, str)):
                # Handle float/string representations like 0.001 or '1e-3'
                precision_float = float(precision_val)
                if 0 < precision_float <= 1:
                    # Calculate decimal places from float like 0.001 -> 3
                    # Use max to handle precision_float == 1 (returns 0)
                    return max(0, abs(int(round(np.log10(precision_float)))))
                elif precision_float > 1:
                    # Handle cases like 10, 100 (meaning round to nearest 10, 100) - convert to negative decimal places
                    return -int(round(np.log10(precision_float)))
                else:  # precision_float is 0 or negative
                    logger.warning(
                        f"Unhandled precision value '{precision_val}' for {precision_type}. Using fallback: {DEFAULT_PRECISION_FALLBACK}"
                    )
                    return DEFAULT_PRECISION_FALLBACK
            else:
                raise TypeError(
                    f"Unexpected type for precision value: {type(precision_val)}"
                )

        except (ValueError, TypeError, OverflowError) as e:
            logger.error(
                f"Could not parse precision value '{precision_val}' for {precision_type}: {e}. Using fallback: {DEFAULT_PRECISION_FALLBACK}"
            )
            return DEFAULT_PRECISION_FALLBACK

    def _format_value(
        self, value: Optional[Union[float, int]], precision_type: str
    ) -> Optional[str]:
        """
        Formats amount or price according to market precision using ccxt helper methods.

        Args:
            value: The numeric value to format.
            precision_type: 'amount' or 'price'.

        Returns:
            The formatted value as a string, or None if formatting fails or input is None.
        """
        if value is None:
            return None
        if precision_type not in ["amount", "price"]:
            logger.error(f"Invalid precision type '{precision_type}' for formatting.")
            return None

        try:
            if precision_type == "amount":
                # Use amount_to_precision which handles different precision modes (decimal, significant) and truncation
                formatted_value = self.exchange.amount_to_precision(self.symbol, value)
            elif precision_type == "price":
                formatted_value = self.exchange.price_to_precision(self.symbol, value)
            else:  # Should be caught above, but as a safeguard
                return None

            # Basic validation of the formatted string
            if formatted_value is None or not isinstance(formatted_value, str):
                raise ValueError(
                    "CCXT precision formatting returned None or non-string."
                )
            # Check if formatting resulted in zero due to truncation and very small input
            if float(formatted_value) == 0 and float(value) != 0:
                logger.warning(
                    f"Formatting {precision_type} {value} resulted in '0' due to precision rules."
                )
                # Depending on context, might want to return None here if zero is invalid

            return formatted_value

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            # Catch potential errors from ccxt methods or float conversion
            logger.error(
                f"Error formatting {precision_type} ({value}) using exchange precision for {self.symbol}: {e}"
            )
            return None  # Return None if formatting fails

    def calculate_order_size(
        self, price: float, volatility: Optional[float]
    ) -> Optional[float]:
        """
        Calculates the order size in the base currency, considering risk percentage,
        balance, volatility adjustment (optional), and exchange minimums/precision.

        Args:
            price: The current market price (used for conversion and min cost check).
            volatility: The calculated volatility (optional, used for size adjustment).

        Returns:
            The calculated order size in base currency (float), adjusted for precision
            and minimums, or None if calculation fails or size is invalid/too small.
        """
        if price <= 0:
            logger.error(f"Invalid price ({price}) for order size calculation.")
            return None

        balance = self.fetch_balance()
        if balance is None or balance <= 0:
            # fetch_balance logs the reason (simulated or live)
            logger.warning(
                f"Cannot calculate order size: Available balance is {balance}."
            )
            return None

        quote_currency = self.market.get("quote", "QUOTE")
        base_currency = self.market.get("base", "BASE")

        # 1. Calculate base size based on risk percentage of balance (in quote currency)
        # Ensure percentage is > 0 to avoid zero size immediately
        if self.order_size_percentage <= 0:
            logger.error("Order size percentage must be positive.")
            return None

        base_size_quote = balance * self.order_size_percentage
        if base_size_quote <= 0:
            logger.warning(
                f"Initial order size based on balance ({balance:.4f} {quote_currency}) and risk percentage ({self.order_size_percentage:.2%}) is zero or negative."
            )
            return None
        logger.debug(
            f"Risk capital allocated: {base_size_quote:.{self._get_precision_digits('price')}f} {quote_currency} ({self.order_size_percentage:.2%} of {balance:.2f})"
        )

        # 2. Adjust size based on volatility (optional)
        adjusted_size_quote = base_size_quote
        if (
            volatility is not None
            and volatility > 1e-9
            and self.volatility_multiplier > 0
        ):
            # Reduce size more in higher volatility; adjustment factor <= 1
            # Simple inverse relationship: size = base_size / (1 + vol * multiplier)
            # Ensure adjustment factor doesn't become zero or negative (volatility should be positive)
            adjustment_factor = max(
                0.01, 1.0 / (1.0 + (volatility * self.volatility_multiplier))
            )  # Add a floor (e.g., 1%)
            adjusted_size_quote = base_size_quote * adjustment_factor
            logger.debug(
                f"Volatility adjustment factor: {adjustment_factor:.4f} (Volatility: {volatility:.5f}, Multiplier: {self.volatility_multiplier})"
            )
            logger.debug(
                f"Adjusted order size (quote): {adjusted_size_quote:.{self._get_precision_digits('price')}f} {quote_currency}"
            )
        else:
            logger.debug(
                "No volatility adjustment applied (Volatility N/A, zero, or multiplier is 0)."
            )

        # 3. Convert quote size to base size
        if price <= 0:  # Double check price just before division
            logger.error("Price is zero or negative before converting to base size.")
            return None
        calculated_size_base = adjusted_size_quote / price
        logger.debug(
            f"Calculated base size before constraints: {calculated_size_base:.{DEFAULT_PRECISION_FALLBACK}f} {base_currency}"
        )

        # 4. Apply exchange constraints (minimums and precision)
        limits = self.market.get("limits", {})
        min_amount = limits.get("amount", {}).get("min")
        min_cost = limits.get("cost", {}).get("min")

        # Convert limits to float for comparison, handle None
        min_amount_float = float(min_amount) if min_amount is not None else None
        min_cost_float = float(min_cost) if min_cost is not None else None

        final_size_base = calculated_size_base

        # --- Check Minimum Cost ---
        # Compare the *intended* quote value (adjusted_size_quote) against min_cost
        if min_cost_float is not None and adjusted_size_quote < min_cost_float:
            logger.warning(
                f"Calculated order cost ({adjusted_size_quote:.{self._get_precision_digits('price')}f} {quote_currency}) "
                f"is below exchange minimum cost ({min_cost_float:.{self._get_precision_digits('price')}f} {quote_currency})."
            )
            # Option 1: Increase size to meet min_cost IF it doesn't exceed original risk budget
            if min_cost_float <= base_size_quote:
                logger.info(
                    f"Attempting to increase size to meet minimum cost ({min_cost_float} {quote_currency})."
                )
                final_size_base = (
                    min_cost_float / price
                )  # Calculate base amount needed for min cost
                logger.debug(
                    f"Adjusted base size to meet min cost: {final_size_base:.{DEFAULT_PRECISION_FALLBACK}f} {base_currency}"
                )
            # Option 2: Fail the trade if min cost exceeds risk budget
            else:
                logger.error(
                    f"Minimum order cost ({min_cost_float} {quote_currency}) exceeds allocated risk capital "
                    f"({base_size_quote:.2f} {quote_currency}). Cannot place trade."
                )
                return None

        # --- Apply Amount Precision ---
        # Apply precision to the potentially adjusted final_size_base
        precise_size_base_str = self._format_value(final_size_base, "amount")
        if precise_size_base_str is None:
            logger.error(
                f"Failed to format order size {final_size_base} according to amount precision. Cannot determine valid order size."
            )
            return None  # Safer to fail if precision cannot be applied

        try:
            precise_size_base = float(precise_size_base_str)
        except ValueError:
            logger.error(
                f"Formatted amount '{precise_size_base_str}' could not be converted back to float."
            )
            return None

        # Check for effective zero after precision formatting
        if precise_size_base <= 1e-12:  # Use a small threshold
            logger.warning(
                f"Order size became zero or negligible ({precise_size_base}) after applying amount precision. Initial base size: {final_size_base:.{DEFAULT_PRECISION_FALLBACK}f}"
            )
            return None

        if precise_size_base != final_size_base:
            # Log the change due to precision application
            amount_prec_digits = self._get_precision_digits("amount")
            logger.debug(
                f"Applied amount precision ({amount_prec_digits} decimals): {precise_size_base:.{amount_prec_digits}f} (from {final_size_base:.{DEFAULT_PRECISION_FALLBACK}f})"
            )

        final_size_base = precise_size_base  # Update with the precise value

        # --- Check Minimum Amount ---
        # Check the *precise* amount against the minimum amount limit
        if min_amount_float is not None and final_size_base < min_amount_float:
            logger.warning(
                f"Calculated order size after precision ({final_size_base:.{self._get_precision_digits('amount')}f} {base_currency}) "
                f"is below exchange minimum amount ({min_amount_float:.{self._get_precision_digits('amount')}f} {base_currency})."
            )
            # Option 1: Increase size to minimum amount IF allowed by risk and cost
            # Calculate the cost of the minimum amount
            min_amount_cost = min_amount_float * price
            if min_amount_cost <= base_size_quote:  # Check against initial risk budget
                logger.info(
                    f"Adjusting order size to exchange minimum amount: {min_amount_float} {base_currency}"
                )
                final_size_base = min_amount_float
                # Re-apply precision to the minimum amount itself, as it might not perfectly match precision rules
                precise_min_amount_str = self._format_value(final_size_base, "amount")
                if precise_min_amount_str:
                    final_size_base = float(precise_min_amount_str)
                    logger.debug(
                        f"Final size after adjusting to precise minimum amount: {final_size_base}"
                    )
                else:
                    logger.error(
                        "Failed to re-apply precision to minimum amount. Aborting size calculation."
                    )
                    return None
            # Option 2: Fail the trade if min amount cost exceeds risk budget
            else:
                logger.error(
                    f"Minimum order amount ({min_amount_float} {base_currency} = ~{min_amount_cost:.2f} {quote_currency}) "
                    f"cost exceeds the allocated risk capital ({base_size_quote:.2f} {quote_currency}). Skipping trade."
                )
                return None

        # --- Final Checks ---
        # Final check for zero or negative size after all adjustments
        if final_size_base <= 1e-12:  # Use a small threshold
            logger.error(
                f"Final calculated order size is zero or negative ({final_size_base}). Skipping trade."
            )
            return None

        # Final check: ensure the final cost doesn't exceed the initial risk capital (due to rounding up for min limits)
        final_cost = final_size_base * price
        if (
            final_cost > base_size_quote * 1.01
        ):  # Allow a small tolerance (e.g., 1%) for rounding
            logger.error(
                f"Final order cost ({final_cost:.2f} {quote_currency}) significantly exceeds allocated risk capital "
                f"({base_size_quote:.2f} {quote_currency}) after adjustments. Aborting trade."
            )
            return None

        # Log the final calculated size
        price_prec_digits = self._get_precision_digits("price")
        amount_prec_digits = self._get_precision_digits("amount")
        logger.info(
            f"Final Calculated Order Size: {final_size_base:.{amount_prec_digits}f} {base_currency} "
            f"(Est. Quote Value: ~{final_cost:.{price_prec_digits}f} {quote_currency})"
        )
        return final_size_base

    # --- Trade Execution Logic ---

    def compute_trade_signal_score(
        self, price: float, indicators: dict, orderbook_imbalance: Optional[float]
    ) -> Tuple[int, List[str]]:
        """
        Computes a simple score based on indicator signals and order book imbalance.
        Higher positive score suggests BUY, higher negative score suggests SELL.

        Args:
            price: Current market price.
            indicators: Dictionary containing calculated indicator values.
            orderbook_imbalance: Calculated bid/ask volume ratio (BidVol / AskVol).

        Returns:
            A tuple containing (score: int, reasons: List[str]).
        """
        score = 0
        reasons = []
        price_prec = self._get_precision_digits("price")

        # --- Extract Indicator Values ---
        # Use .get() to handle potential None values gracefully if calculation failed
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
        # Imbalance threshold (buy if ratio > threshold) is already in self.imbalance_threshold
        # Define sell threshold based on inverse (buy if ratio < sell_threshold)
        # Ensure self.imbalance_threshold is positive to avoid division by zero or weirdness
        sell_imbalance_threshold = (
            (1.0 / self.imbalance_threshold) if self.imbalance_threshold > 1e-9 else 0.0
        )

        # --- Scoring Logic ---

        # 1. Order Book Imbalance (Weight: +/- 1)
        if orderbook_imbalance is not None:
            if orderbook_imbalance > self.imbalance_threshold:
                score += 1
                reasons.append(
                    f"OB Imbalance: Strong bid pressure (Ratio: {orderbook_imbalance:.2f} > {self.imbalance_threshold:.2f})"
                )
            elif orderbook_imbalance < sell_imbalance_threshold:
                score -= 1
                reasons.append(
                    f"OB Imbalance: Strong ask pressure (Ratio: {orderbook_imbalance:.2f} < {sell_imbalance_threshold:.2f})"
                )
            else:
                reasons.append(
                    f"OB Imbalance: Neutral pressure (Ratio: {orderbook_imbalance:.2f})"
                )
        else:
            reasons.append("OB Imbalance: N/A")

        # 2. EMA Trend (Weight: +/- 1)
        if ema is not None:
            if price > ema:
                score += 1
                reasons.append(
                    f"Trend: Price > EMA ({price:.{price_prec}f} > {ema:.{price_prec}f}) (Bullish)"
                )
            elif price < ema:
                score -= 1
                reasons.append(
                    f"Trend: Price < EMA ({price:.{price_prec}f} < {ema:.{price_prec}f}) (Bearish)"
                )
            else:
                reasons.append(f"Trend: Price == EMA ({price:.{price_prec}f})")
        else:
            reasons.append("Trend: EMA N/A")

        # 3. RSI Momentum (Weight: +/- 1) - Oversold = Buy signal, Overbought = Sell signal
        if rsi is not None:
            if rsi < rsi_oversold:
                score += 1
                reasons.append(
                    f"Momentum: RSI < {rsi_oversold} ({rsi:.2f}) (Oversold - Potential Buy)"
                )
            elif rsi > rsi_overbought:
                score -= 1
                reasons.append(
                    f"Momentum: RSI > {rsi_overbought} ({rsi:.2f}) (Overbought - Potential Sell)"
                )
            else:
                reasons.append(f"Momentum: RSI Neutral ({rsi:.2f})")
        else:
            reasons.append("Momentum: RSI N/A")

        # 4. MACD Momentum (Weight: +/- 1) - MACD Line vs Signal Line
        if macd_line is not None and macd_signal is not None:
            # Could add check for previous state to confirm *crossover* vs just state
            if macd_line > macd_signal:
                score += 1  # Bullish state/crossover
                reasons.append(
                    f"Momentum: MACD > Signal ({macd_line:.4f} > {macd_signal:.4f}) (Bullish)"
                )
            elif macd_line < macd_signal:
                score -= 1  # Bearish state/crossover
                reasons.append(
                    f"Momentum: MACD < Signal ({macd_line:.4f} < {macd_signal:.4f}) (Bearish)"
                )
            else:
                reasons.append(f"Momentum: MACD == Signal ({macd_line:.4f})")
        else:
            reasons.append("Momentum: MACD N/A")

        # 5. Stochastic RSI Momentum (Weight: +/- 1) - K and D lines in extreme zones
        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            # Simple check: both lines in extreme zones suggest stronger signal
            if stoch_rsi_k < stoch_rsi_oversold and stoch_rsi_d < stoch_rsi_oversold:
                score += 1  # Oversold
                reasons.append(
                    f"Momentum: Stoch RSI < {stoch_rsi_oversold} (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (Oversold - Potential Buy)"
                )
            elif (
                stoch_rsi_k > stoch_rsi_overbought
                and stoch_rsi_d > stoch_rsi_overbought
            ):
                score -= 1  # Overbought
                reasons.append(
                    f"Momentum: Stoch RSI > {stoch_rsi_overbought} (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f}) (Overbought - Potential Sell)"
                )
            # Optional: Add crossover logic (e.g., K crossing above D in oversold zone) for finer signals
            # elif stoch_rsi_k > stoch_rsi_d and stoch_rsi_k < 50: # Example bullish crossover below midline
            #     score += 1 # Add less weight for crossover vs extreme zone?
            #     reasons.append(f"Momentum: Stoch RSI K crossed above D (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f})")
            else:
                reasons.append(
                    f"Momentum: Stoch RSI Neutral (K:{stoch_rsi_k:.2f}, D:{stoch_rsi_d:.2f})"
                )
        else:
            reasons.append("Momentum: Stoch RSI N/A")

        logger.debug(f"Signal Score: {score}, Reasons: {'; '.join(reasons)}")
        return score, reasons

    @retry_api_call(
        max_retries=1, allowed_exceptions=(ccxt.NetworkError, ccxt.RateLimitExceeded)
    )  # Only retry network/rate errors once for orders
    def place_order(
        self,
        side: OrderSide,
        order_size: float,  # Base currency amount
        order_type: OrderType,
        price: Optional[
            float
        ] = None,  # Required for limit orders (quote currency price)
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Places a trade order (simulated or live) with optional SL/TP.

        Handles precision formatting and basic parameter preparation for SL/TP.

        Args:
            side: OrderSide.BUY or OrderSide.SELL.
            order_size: The amount of base currency to trade.
            order_type: OrderType.MARKET or OrderType.LIMIT.
            price: The limit price for LIMIT orders (in quote currency).
            stop_loss_price: Optional stop-loss trigger price (in quote currency).
            take_profit_price: Optional take-profit trigger price (in quote currency).

        Returns:
            The order dictionary returned by CCXT (or a simulated one), or None on failure.
            The dictionary structure may vary between live and simulated orders.
        """
        # --- Input Validation ---
        if not isinstance(side, OrderSide):
            logger.error(f"Invalid order side type: {type(side)}. Use OrderSide enum.")
            return None
        if not isinstance(order_type, OrderType):
            logger.error(
                f"Invalid order type type: {type(order_type)}. Use OrderType enum."
            )
            return None
        if order_type == OrderType.LIMIT and (price is None or price <= 0):
            logger.error(
                f"Valid positive price is required for {OrderType.LIMIT.value} orders, got {price}."
            )
            return None
        if order_size <= 0:
            logger.error(f"Order size must be positive, got {order_size}.")
            return None
        # Validate SL/TP prices if provided
        if stop_loss_price is not None and stop_loss_price <= 0:
            logger.warning(
                f"Stop loss price must be positive, got {stop_loss_price}. Ignoring SL."
            )
            stop_loss_price = None
        if take_profit_price is not None and take_profit_price <= 0:
            logger.warning(
                f"Take profit price must be positive, got {take_profit_price}. Ignoring TP."
            )
            take_profit_price = None

        # --- Format Parameters using Exchange Precision ---
        order_size_str = self._format_value(order_size, "amount")
        price_str = (
            self._format_value(price, "price")
            if order_type == OrderType.LIMIT
            else None
        )
        sl_price_str = (
            self._format_value(stop_loss_price, "price") if stop_loss_price else None
        )
        tp_price_str = (
            self._format_value(take_profit_price, "price")
            if take_profit_price
            else None
        )

        # --- Critical Formatting Checks ---
        # Fail if essential parameters cannot be formatted correctly
        if order_size_str is None or float(order_size_str) <= 0:
            logger.error(
                f"Failed to format order size {order_size} or result is zero/negative. Cannot place order."
            )
            return None
        if order_type == OrderType.LIMIT and price_str is None:
            logger.error(
                f"Failed to format limit order price {price}. Cannot place order."
            )
            return None

        # Log warnings if optional SL/TP formatting fails, but proceed without them
        if stop_loss_price and sl_price_str is None:
            logger.warning(
                f"Failed to format stop loss price {stop_loss_price}, SL will not be attached."
            )
        if take_profit_price and tp_price_str is None:
            logger.warning(
                f"Failed to format take profit price {take_profit_price}, TP will not be attached."
            )

        # Convert formatted strings back to float where needed by CCXT methods (e.g., for limit price)
        # Use the precise string versions for parameters where possible
        order_size_float = float(order_size_str)  # Keep float for simulation logic
        price_float = float(price_str) if price_str else None

        # --- Prepare CCXT `params` for SL/TP ---
        # This is highly exchange-specific. Use unified params where possible,
        # but add specific examples and acknowledge limitations.
        params = {}
        sl_tp_info_log = ""  # For logging purposes

        # Stop Loss
        if sl_price_str:
            sl_tp_info_log += f" SL:{sl_price_str}"
            # Unified parameter (check exchange.has['stopLossPrice'])
            if self.exchange.has.get("stopLossPrice"):
                params["stopLossPrice"] = sl_price_str  # Trigger price
                # params['stopLoss'] = {'triggerPrice': sl_price_str, 'type': 'market'} # More complex structure if needed
                logger.debug(f"Using unified 'stopLossPrice' param: {sl_price_str}")
            # Exchange-specific examples (add more as needed based on testing)
            elif self.exchange.id in ["binance", "binanceusdm", "binancecoinm"]:
                params["stopPrice"] = sl_price_str  # Binance uses stopPrice for trigger
                params["type"] = (
                    "STOP_MARKET"  # Or TAKE_PROFIT_MARKET if it's a TP order being placed as a stop
                )
                logger.debug(
                    f"Using Binance specific SL params: stopPrice={sl_price_str}, type=STOP_MARKET"
                )
            elif self.exchange.id == "bybit":
                params["stopLoss"] = (
                    sl_price_str  # Bybit often uses direct price string in unified field
                )
                # params['slTriggerBy'] = 'LastPrice' # Example: specify trigger type if needed
                logger.debug(f"Using Bybit specific SL param: stopLoss={sl_price_str}")
            # Add other exchanges here...
            else:
                logger.warning(
                    f"Stop Loss parameter structure for {self.exchange_id} is not explicitly defined. Attempting generic 'stopLossPrice'. Order might fail or SL might not be set."
                )
                params["stopLossPrice"] = sl_price_str  # Fallback attempt

        # Take Profit
        if tp_price_str:
            sl_tp_info_log += f" TP:{tp_price_str}"
            # Unified parameter (check exchange.has['takeProfitPrice'])
            if self.exchange.has.get("takeProfitPrice"):
                params["takeProfitPrice"] = tp_price_str  # Trigger price
                logger.debug(f"Using unified 'takeProfitPrice' param: {tp_price_str}")
            # Exchange-specific examples
            elif self.exchange.id in ["binance", "binanceusdm", "binancecoinm"]:
                # Binance uses stopPrice for TP triggers too, with a different type
                # Note: Cannot set SL and TP simultaneously using simple stopPrice on Binance createOrder
                if "stopPrice" in params:
                    logger.warning(
                        "Cannot set both SL and TP using 'stopPrice' on Binance during initial order placement. TP ignored."
                    )
                else:
                    params["stopPrice"] = tp_price_str
                    params["type"] = "TAKE_PROFIT_MARKET"  # Or LIMIT if needed
                    logger.debug(
                        f"Using Binance specific TP params: stopPrice={tp_price_str}, type=TAKE_PROFIT_MARKET"
                    )
            elif self.exchange.id == "bybit":
                params["takeProfit"] = tp_price_str
                # params['tpTriggerBy'] = 'LastPrice'
                logger.debug(
                    f"Using Bybit specific TP param: takeProfit={tp_price_str}"
                )
            # Add other exchanges here...
            else:
                logger.warning(
                    f"Take Profit parameter structure for {self.exchange_id} is not explicitly defined. Attempting generic 'takeProfitPrice'. Order might fail or TP might not be set."
                )
                params["takeProfitPrice"] = tp_price_str  # Fallback attempt

        # Log the final parameters being sent (be careful with sensitive info if any)
        logger.debug(f"Prepared CCXT order params: {params}")

        # --- Simulation Mode ---
        if self.simulation_mode:
            quote_currency = self.market.get("quote", "QUOTE")
            base_currency = self.market.get("base", "BASE")
            price_prec = self._get_precision_digits("price")
            amount_prec = self._get_precision_digits("amount")

            # Determine simulated fill price
            simulated_fill_price = None
            current_market_price = (
                self.fetch_market_price()
            )  # Fetch current price for simulation logic
            if current_market_price is None:
                logger.error(
                    "[SIMULATION] Could not fetch market price to evaluate order fill. Aborting."
                )
                return None

            if order_type == OrderType.LIMIT:
                # Simulate fill only if market price crosses the limit price (simplistic)
                # Assume immediate fill if condition met for simplicity
                if side == OrderSide.BUY and current_market_price <= price_float:
                    simulated_fill_price = price_float  # Fill at limit price
                elif side == OrderSide.SELL and current_market_price >= price_float:
                    simulated_fill_price = price_float  # Fill at limit price
                else:
                    logger.info(
                        f"[SIMULATION] Limit order ({side.value} @ {price_float:.{price_prec}f}) "
                        f"NOT filled at current market price ({current_market_price:.{price_prec}f})."
                    )
                    # Return a structure indicating an open (unfilled) order? Or None?
                    # For this simulation, assume we only care about filled orders. Return None.
                    # TODO: Could enhance simulation to track open limit orders.
                    return None  # Indicate order not filled
            else:  # Market order
                # Simulate fill at the fetched current market price (could add slippage simulation later)
                simulated_fill_price = current_market_price

            simulated_fill_price = float(simulated_fill_price)  # Ensure float

            # Calculate cost and fee
            cost = order_size_float * simulated_fill_price
            fee = cost * self.simulated_fee_rate
            simulated_cost_with_fee = (
                cost + fee if side == OrderSide.BUY else cost - fee
            )
            balance_change = (
                -simulated_cost_with_fee if side == OrderSide.BUY else cost - fee
            )  # Payout for sell

            # Update simulated balance
            self.simulated_balance_current += balance_change
            logger.info(
                f"[SIMULATION] Balance updated: {self.simulated_balance_current:.{price_prec}f} {quote_currency} (Change: {balance_change:.{price_prec}f})"
            )

            # Create a simulated order dictionary mimicking CCXT structure
            trade_details = {
                "id": f"sim_{int(time.time() * 1000)}_{side.value[:1]}",
                "symbol": self.symbol,
                "status": "closed",  # Simulate immediate fill and close
                "side": side.value,
                "type": order_type.value,
                "amount": order_size_float,  # Requested amount
                "price": price_float
                if order_type == OrderType.LIMIT
                else None,  # Requested limit price
                "average": simulated_fill_price,  # Actual fill price
                "cost": abs(
                    simulated_cost_with_fee
                ),  # Total cost/proceeds including fee
                "filled": order_size_float,  # Assume fully filled
                "remaining": 0.0,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.Timestamp.now(tz="UTC").isoformat(),
                "fee": {"cost": fee, "currency": quote_currency},
                "info": {  # Include intended SL/TP and simulation flag
                    "simulated_stopLoss": sl_price_str,
                    "simulated_takeProfit": tp_price_str,
                    "orderType": order_type.value.capitalize(),
                    "execType": "Trade",
                    "is_simulated": True,
                },
            }
            log_price_req = (
                f" @ {price_str}" if order_type == OrderType.LIMIT else " Market"
            )
            logger.info(
                f"{Fore.YELLOW}[SIMULATION] Order Filled: ID: {trade_details['id']}, Type: {trade_details['type']}{log_price_req}, "
                f"Side: {trade_details['side'].upper()}, Size: {trade_details['amount']:.{amount_prec}f} {base_currency}, "
                f"Fill Price: {simulated_fill_price:.{price_prec}f}, {sl_tp_info_log}, "
                f"Simulated Cost: {trade_details['cost']:.{price_prec}f} {quote_currency} (Fee: {fee:.{price_prec + 2}f})"
            )
            return trade_details

        # --- Live Trading Mode ---
        else:
            order = None
            order_action = f"{side.value.upper()} {order_type.value.upper()}"
            price_log = f" @ {price_str}" if order_type == OrderType.LIMIT else ""
            order_details_log = f"{order_size_str} {self.market.get('base', 'BASE')}{price_log}{sl_tp_info_log}"

            try:
                logger.info(
                    f"{Fore.CYAN}Placing LIVE {order_action} order for {order_details_log} (Params: {params})"
                )

                # Use create_order for flexibility with params across exchanges
                # Pass amount and price as strings formatted by _format_value where appropriate
                # CCXT generally prefers strings for precision, but check specific method docs if issues arise
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type=order_type.value,  # 'market' or 'limit'
                    side=side.value,  # 'buy' or 'sell'
                    amount=order_size_float,  # Pass float amount, CCXT handles internal conversion/formatting
                    price=price_float
                    if order_type == OrderType.LIMIT
                    else None,  # Pass float price for limit
                    params=params,  # Pass SL/TP and other specific params here
                )

                if order and isinstance(order, dict):
                    # Log details from the returned order structure
                    order_id = order.get("id", "N/A")
                    order_status = order.get("status", "N/A")
                    avg_price = order.get("average")
                    filled_amount = order.get("filled", 0.0)
                    cost = order.get(
                        "cost"
                    )  # Cost might include fees depending on exchange

                    # Determine price to display in log (average if filled, otherwise requested price)
                    log_price_actual = (
                        avg_price if avg_price is not None else order.get("price")
                    )
                    price_prec = self._get_precision_digits("price")
                    amount_prec = self._get_precision_digits("amount")
                    log_price_display = (
                        f"{log_price_actual:.{price_prec}f}"
                        if log_price_actual
                        else (price_str if order_type == OrderType.LIMIT else "Market")
                    )
                    cost_display = (
                        f"{cost:.{price_prec}f} {self.market.get('quote', 'QUOTE')}"
                        if cost is not None
                        else "N/A"
                    )

                    logger.info(
                        f"{Fore.GREEN}LIVE Order Response: ID: {order_id}, Type: {order.get('type')}, Side: {order.get('side', '').upper()}, "
                        f"Amount: {order.get('amount'):.{amount_prec}f}, Filled: {filled_amount:.{amount_prec}f} {self.market.get('base', 'BASE')}, "
                        f"AvgPrice: {log_price_display}, Cost: {cost_display}, "
                        # Note: Params sent might not reflect actual state if exchange rejected/modified SL/TP
                        f"Status: {order_status}"
                    )
                    # TODO: Add robust tracking of the actual order state by fetching order status later if needed,
                    # especially if status is 'open' or SL/TP were involved.
                    # self.open_positions.append(order) # Manage this in the main loop or a dedicated manager
                    return order
                else:
                    # This case might occur if the API call succeeded (no exception) but CCXT returned None/empty/unexpected
                    logger.error(
                        f"LIVE Order placement call for {order_action} succeeded but returned unexpected result: {order}"
                    )
                    return None

            # --- Specific CCXT Exception Handling for Orders ---
            # These might not be retried by the decorator depending on its config
            except ccxt.InsufficientFunds as e:
                logger.error(
                    f"{Fore.RED}LIVE Order Failed: Insufficient funds for {order_action} {order_details_log}. Error: {e}"
                )
                return None  # Definite failure, don't retry
            except ccxt.InvalidOrder as e:
                logger.error(
                    f"{Fore.RED}LIVE Order Failed: Invalid order parameters for {order_action} {order_details_log}. Check size, price, limits, params. Error: {e}"
                )
                return None  # Definite failure, don't retry
            except ccxt.ExchangeError as e:
                # Catch other exchange errors that might occur during placement (e.g., margin errors, specific rejections)
                logger.error(
                    f"{Fore.RED}LIVE Order Failed: Exchange error during placement of {order_action} {order_details_log}. Error: {e}"
                )
                # Check if retry decorator handled this type, if not return None
                return None  # Assume failure if not retried
            # NetworkError and RateLimitExceeded should be handled by the decorator if configured
            except Exception:
                # Catch any unexpected Python errors during the process
                logger.exception(
                    f"{Fore.RED}LIVE Order Failed: An unexpected error occurred during order placement of {order_action} {order_details_log}."
                )
                return None

    # --- Position Management (Basic Structure) ---
    # TODO: Implement robust position management, potentially in a separate class.
    # This requires fetching open orders/positions from the exchange, reconciling
    # with internal state, handling partial fills, and managing SL/TP logic post-placement.

    def _update_position_state(
        self,
        order_result: Dict[str, Any],
        side: OrderSide,
        sl_price: Optional[float],
        tp_price: Optional[float],
    ):
        """
        Updates the internal state (self.open_positions) based on a new order result.
        This is a basic implementation for demonstration.
        """
        if not order_result or not isinstance(order_result, dict):
            return

        order_id = order_result.get("id")
        status = order_result.get("status")
        filled_amount = order_result.get("filled", 0.0)
        entry_price = order_result.get("average") or order_result.get(
            "price"
        )  # Use average if available

        if not order_id or filled_amount <= 0 or entry_price is None:
            logger.warning(
                f"Order result lacks necessary info to track position: {order_result}"
            )
            return

        # Simple tracking: Add to list if filled and status suggests it's active ('open' or 'closed' for immediate fills)
        # A real system needs more complex state (open, partially filled, closed, cancelled)
        if (
            status in ["closed", "open"] and filled_amount > 0
        ):  # 'closed' for market/simulated, 'open' for live limit
            position_data = {
                "id": order_id,
                "symbol": self.symbol,
                "side": side,
                "entry_price": float(entry_price),
                "size": float(filled_amount),  # Track filled amount
                "status": status,  # Store initial status
                "entry_time": order_result.get("timestamp", int(time.time() * 1000)),
                "sl_intended": sl_price,  # Store intended SL/TP (actual might differ)
                "tp_intended": tp_price,
                "is_simulated": order_result.get("info", {}).get("is_simulated", False),
                # Add fields needed for trailing SL etc. later
                "highest_price_since_entry": float(entry_price)
                if side == OrderSide.BUY
                else float("-inf"),
                "lowest_price_since_entry": float(entry_price)
                if side == OrderSide.SELL
                else float("inf"),
                "trailing_sl_price": None,  # Activated trailing stop price
            }
            self.open_positions.append(position_data)
            logger.info(
                f"Tracking new position: ID={order_id}, Side={side.value}, Size={position_data['size']}, Entry={position_data['entry_price']}"
            )
        else:
            logger.info(
                f"Order {order_id} not added to tracked positions (Status: {status}, Filled: {filled_amount})"
            )

    def manage_open_positions(self, current_price: float):
        """
        Manages currently tracked open positions.

        Checks for Stop Loss, Take Profit, Time-based exits, and updates Trailing Stops.
        Places closing orders if exit conditions are met.

        NOTE: This is a basic implementation using an internal list (`self.open_positions`).
        A robust system MUST fetch actual order/position status from the exchange
        and reconcile it with the internal state. This implementation assumes the
        internal list accurately reflects open positions and doesn't handle partial fills
        or exchange-side closures adequately.
        """
        if not self.open_positions:
            logger.debug("No open positions to manage.")
            return

        logger.debug(
            f"Managing {len(self.open_positions)} tracked position(s) at current price {current_price:.{self._get_precision_digits('price')}f}..."
        )

        positions_to_remove_indices = []  # Store indices to remove later

        for i, pos in enumerate(self.open_positions):
            pos_id = pos.get("id", "N/A")
            pos_side = pos.get("side")
            pos_size = pos.get("size", 0)
            entry_price = pos.get("entry_price")
            sl_intended = pos.get("sl_intended")
            tp_intended = pos.get("tp_intended")
            entry_time_ms = pos.get("entry_time")
            is_simulated = pos.get("is_simulated", False)

            if not all([pos_id, pos_side, pos_size > 0, entry_price]):
                logger.warning(
                    f"Skipping invalid position data in tracking list: {pos}"
                )
                continue

            logger.debug(
                f"Checking position {pos_id}: Side={pos_side.value}, Size={pos_size}, Entry={entry_price}, SL={sl_intended}, TP={tp_intended}"
            )

            exit_reason = None
            close_position = False

            # --- Check Exit Conditions ---
            # 1. Stop Loss (using intended SL for simplicity)
            if sl_intended:
                if pos_side == OrderSide.BUY and current_price <= sl_intended:
                    exit_reason = f"Stop Loss triggered ({current_price:.{self._get_precision_digits('price')}f} <= {sl_intended:.{self._get_precision_digits('price')}f})"
                    close_position = True
                elif pos_side == OrderSide.SELL and current_price >= sl_intended:
                    exit_reason = f"Stop Loss triggered ({current_price:.{self._get_precision_digits('price')}f} >= {sl_intended:.{self._get_precision_digits('price')}f})"
                    close_position = True

            # 2. Take Profit (only if SL not already triggered)
            if not close_position and tp_intended:
                if pos_side == OrderSide.BUY and current_price >= tp_intended:
                    exit_reason = f"Take Profit triggered ({current_price:.{self._get_precision_digits('price')}f} >= {tp_intended:.{self._get_precision_digits('price')}f})"
                    close_position = True
                elif pos_side == OrderSide.SELL and current_price <= tp_intended:
                    exit_reason = f"Take Profit triggered ({current_price:.{self._get_precision_digits('price')}f} <= {tp_intended:.{self._get_precision_digits('price')}f})"
                    close_position = True

            # 3. Time-based Exit (only if SL/TP not already triggered)
            if (
                not close_position
                and self.time_based_exit_minutes > 0
                and entry_time_ms
            ):
                current_time_ms = time.time() * 1000
                duration_minutes = (current_time_ms - entry_time_ms) / (1000 * 60)
                if duration_minutes >= self.time_based_exit_minutes:
                    exit_reason = f"Time-based exit triggered ({duration_minutes:.1f} >= {self.time_based_exit_minutes} minutes)"
                    close_position = True

            # 4. Trailing Stop Loss (Needs more state: activation price, highest/lowest price)
            # TODO: Implement Trailing Stop Loss logic if self.trailing_stop_loss_percentage > 0
            # - Update highest/lowest price since entry
            # - Calculate potential new SL based on trailing percentage
            # - Update pos['trailing_sl_price'] if it improves
            # - Check if current_price hits pos['trailing_sl_price']

            # --- Execute Closing Order ---
            if close_position:
                logger.info(
                    f"{Fore.MAGENTA}Exit condition met for position {pos_id}: {exit_reason}. Attempting to close."
                )
                close_side = (
                    OrderSide.SELL if pos_side == OrderSide.BUY else OrderSide.BUY
                )

                # Place MARKET order to close the position
                # Use the tracked size. A real system needs to confirm actual open size.
                close_order_result = self.place_order(
                    side=close_side,
                    order_size=pos_size,
                    order_type=OrderType.MARKET,
                    # No SL/TP needed for closing order
                )

                if close_order_result:
                    logger.info(
                        f"{Fore.GREEN}Successfully placed closing order for position {pos_id}. Result: {close_order_result.get('id')}"
                    )
                    # Mark position for removal from internal tracking
                    positions_to_remove_indices.append(i)

                    # --- Simulate PnL Update ---
                    if is_simulated:
                        exit_price = close_order_result.get(
                            "average", current_price
                        )  # Use fill price if available
                        pnl_per_unit = (
                            (exit_price - entry_price)
                            if pos_side == OrderSide.BUY
                            else (entry_price - exit_price)
                        )
                        gross_pnl = pnl_per_unit * pos_size
                        # Subtract simulated fees for both entry and exit
                        entry_cost = entry_price * pos_size
                        exit_cost = exit_price * pos_size
                        total_fees = (entry_cost * self.simulated_fee_rate) + (
                            exit_cost * self.simulated_fee_rate
                        )
                        net_pnl = gross_pnl - total_fees
                        self.daily_pnl += (
                            net_pnl  # Accumulate PnL (needs reset logic daily/session)
                        )
                        logger.info(
                            f"[SIMULATION] Position {pos_id} closed. Est. Net PnL: {net_pnl:.{self._get_precision_digits('price')}f} {self.market['quote']} (Gross: {gross_pnl:.{self._get_precision_digits('price')}f}, Fees: {total_fees:.{self._get_precision_digits('price')}f})"
                        )
                        logger.info(
                            f"[SIMULATION] Cumulative Session PnL: {self.daily_pnl:.{self._get_precision_digits('price')}f}"
                        )

                else:
                    # Critical: Failed to close position! Requires manual intervention or retry logic.
                    logger.error(
                        f"{Fore.RED}CRITICAL: Failed to place closing order for position {pos_id} which met exit condition ({exit_reason}). Manual intervention may be required."
                    )
                    # Keep position in list for now, maybe retry closing later?

        # --- Remove Closed Positions from Tracking ---
        # Remove indices in reverse order to avoid shifting issues
        if positions_to_remove_indices:
            logger.debug(
                f"Removing {len(positions_to_remove_indices)} closed positions from tracking list."
            )
            for index in sorted(positions_to_remove_indices, reverse=True):
                try:
                    removed_pos = self.open_positions.pop(index)
                    logger.debug(
                        f"Removed position {removed_pos.get('id')} from tracking."
                    )
                except IndexError:
                    logger.error(
                        f"Error removing position at index {index}, list may have changed unexpectedly."
                    )

    # --- Main Loop Logic ---

    def run(self):
        """Main execution loop for the bot."""
        logger.info(f"{Fore.CYAN}==================================================")
        logger.info(f"{Fore.CYAN} Starting Scalping Bot")
        logger.info(f"{Fore.CYAN} Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        logger.info(f"{Fore.CYAN} Loop Delay: {self.trade_loop_delay}s")
        if self.simulation_mode:
            logger.warning(f"{Fore.YELLOW}--- RUNNING IN SIMULATION MODE ---")
            logger.info(
                f"Initial Simulated Balance: {self.simulated_balance:.{self._get_precision_digits('price')}f} {self.market.get('quote', 'QUOTE')}, Fee Rate: {self.simulated_fee_rate:.3%}"
            )
        else:
            logger.warning(f"{Fore.RED}--- RUNNING IN LIVE TRADING MODE ---")
        logger.info(f"{Fore.CYAN}==================================================")

        while True:
            self.iteration += 1
            start_time = (
                time.monotonic()
            )  # Use monotonic clock for measuring intervals accurately
            timestamp_now = pd.Timestamp.now(tz="UTC")
            logger.info(
                f"\n----- Iteration {self.iteration} | Time: {timestamp_now.isoformat()} -----"
            )

            try:
                # --- 1. Fetch Required Data ---
                # Use asyncio gather in future for concurrent fetches if switching to async framework
                current_price = self.fetch_market_price()
                orderbook_imbalance = self.fetch_order_book()
                # Fetch historical data - recalculates required limit each time based on config
                historical_data = self.fetch_historical_data()

                # --- Basic Data Validation ---
                if current_price is None:
                    logger.warning(
                        "Could not fetch current market price. Skipping iteration."
                    )
                    self._wait_for_next_iteration(start_time)
                    continue
                if historical_data is None:
                    logger.warning(
                        "Could not fetch valid historical data. Skipping iteration."
                    )
                    self._wait_for_next_iteration(start_time)
                    continue

                # --- Check for New Candle Data (Optional Optimization) ---
                # Avoid redundant calculations if loop runs faster than timeframe
                current_candle_ts_ms = (
                    historical_data.index[-1].value // 10**6
                )  # Timestamp in milliseconds
                if (
                    self.last_candle_ts is not None
                    and current_candle_ts_ms <= self.last_candle_ts
                ):
                    # Check if enough time has passed anyway (e.g., half the loop delay) to force update
                    time_since_last_candle = time.monotonic() - getattr(
                        self, "_last_candle_update_time", start_time
                    )
                    if time_since_last_candle < self.trade_loop_delay / 2:
                        logger.debug(
                            f"No new candle data since last check ({pd.Timestamp(self.last_candle_ts, unit='ms', tz='UTC')}). Waiting..."
                        )
                        self._wait_for_next_iteration(start_time)
                        continue
                    else:
                        logger.debug(
                            "Forcing update even without new candle due to time elapsed."
                        )
                logger.debug(f"Processing data for candle: {historical_data.index[-1]}")
                self.last_candle_ts = current_candle_ts_ms
                setattr(
                    self, "_last_candle_update_time", time.monotonic()
                )  # Track time of update

                # --- 2. Calculate Indicators ---
                # Pass the fetched dataframe to calculation methods
                volatility = self.calculate_volatility(historical_data)
                ema = self.calculate_ema(historical_data, self.ema_period)
                rsi = self.calculate_rsi(historical_data)
                macd_line, macd_signal, macd_hist = self.calculate_macd(historical_data)
                stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(historical_data)

                indicators = {
                    "volatility": volatility,
                    "ema": ema,
                    "rsi": rsi,
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "macd_hist": macd_hist,
                    "stoch_rsi_k": stoch_rsi_k,
                    "stoch_rsi_d": stoch_rsi_d,
                }
                # Check if any critical indicator failed calculation (None value)
                # Define which indicators are critical for signal generation
                critical_indicators = [
                    "ema",
                    "rsi",
                    "macd_line",
                    "macd_signal",
                    "stoch_rsi_k",
                    "stoch_rsi_d",
                ]
                if any(indicators.get(k) is None for k in critical_indicators):
                    logger.warning(
                        "One or more critical indicators failed calculation. Skipping signal generation and position management for this iteration."
                    )
                    self._wait_for_next_iteration(start_time)
                    continue

                # --- 3. Manage Open Positions (Check for Exits: SL/TP/Trailing/Time) ---
                # This needs to be implemented properly based on actual position tracking
                # Call *before* checking entry conditions
                self.manage_open_positions(current_price)

                # --- 4. Generate Trade Signal (Entry Logic) ---
                # Check if max positions limit reached *after* managing exits
                current_open_count = len(
                    self.open_positions
                )  # Based on internal tracking list
                if current_open_count >= self.max_open_positions:
                    logger.info(
                        f"Max open positions ({self.max_open_positions}) reached. Holding. Currently open: {current_open_count}"
                    )
                else:
                    logger.debug(
                        f"Open positions ({current_open_count}) below limit ({self.max_open_positions}). Checking for entry signals..."
                    )
                    score, reasons = self.compute_trade_signal_score(
                        current_price, indicators, orderbook_imbalance
                    )

                    # Define entry thresholds (example: require a strong signal, make configurable?)
                    # These thresholds determine how many indicators must agree.
                    buy_threshold = 2  # Example: Need score >= 2 for BUY
                    sell_threshold = -2  # Example: Need score <= -2 for SELL
                    trade_side: Optional[OrderSide] = None
                    signal_color = Style.RESET_ALL

                    if score >= buy_threshold:
                        trade_side = OrderSide.BUY
                        signal_color = Fore.GREEN
                        logger.info(
                            f"{signal_color}Potential BUY signal (Score: {score}). Reasons: {'; '.join(reasons)}"
                        )
                    elif score <= sell_threshold:
                        trade_side = OrderSide.SELL
                        signal_color = Fore.RED
                        logger.info(
                            f"{signal_color}Potential SELL signal (Score: {score}). Reasons: {'; '.join(reasons)}"
                        )
                    else:
                        logger.info(f"Neutral signal (Score: {score}). Holding.")

                    # --- 5. Place Order if Signal Found and Position Limit Not Reached ---
                    if trade_side:
                        logger.info(
                            f"{signal_color}Attempting to place {trade_side.value.upper()} order based on signal score {score}."
                        )
                        # Calculate order size based on current price and risk settings
                        order_size = self.calculate_order_size(
                            current_price, volatility
                        )

                        if order_size is not None and order_size > 0:
                            # Calculate SL/TP prices based on current price and configured percentages
                            sl_price, tp_price = None, None
                            price_prec = self._get_precision_digits("price")

                            if trade_side == OrderSide.BUY:
                                if self.stop_loss_pct > 0:
                                    sl_price = current_price * (1 - self.stop_loss_pct)
                                if self.take_profit_pct > 0:
                                    tp_price = current_price * (
                                        1 + self.take_profit_pct
                                    )
                                # Adjust limit order entry price slightly below current for buys to improve fill chance
                                entry_price = (
                                    current_price
                                    * (1 - self.limit_order_offset_buy_pct)
                                    if self.entry_order_type == OrderType.LIMIT
                                    else None
                                )
                            else:  # SELL
                                if self.stop_loss_pct > 0:
                                    sl_price = current_price * (1 + self.stop_loss_pct)
                                if self.take_profit_pct > 0:
                                    tp_price = current_price * (
                                        1 - self.take_profit_pct
                                    )
                                # Adjust limit order entry price slightly above current for sells
                                entry_price = (
                                    current_price
                                    * (1 + self.limit_order_offset_sell_pct)
                                    if self.entry_order_type == OrderType.LIMIT
                                    else None
                                )

                            # Log calculated prices before formatting/placement
                            logger.debug(
                                f"Calculated Entry: {entry_price:.{price_prec}f if entry_price else 'Market'}, "
                                f"SL: {sl_price:.{price_prec}f if sl_price else 'N/A'}, "
                                f"TP: {tp_price:.{price_prec}f if tp_price else 'N/A'} (before precision)"
                            )

                            # Place the order (handles simulation vs live, precision formatting)
                            order_result = self.place_order(
                                side=trade_side,
                                order_size=order_size,
                                order_type=self.entry_order_type,
                                price=entry_price,  # None for market orders
                                stop_loss_price=sl_price,
                                take_profit_price=tp_price,
                            )

                            if order_result:
                                logger.info(
                                    f"{signal_color}Successfully processed {trade_side.value.upper()} order placement request."
                                )
                                # Update internal position tracking (basic implementation)
                                self._update_position_state(
                                    order_result, trade_side, sl_price, tp_price
                                )
                            else:
                                logger.error(
                                    f"{signal_color}Failed to place {trade_side.value.upper()} order after signal."
                                )
                        else:
                            logger.warning(
                                f"{signal_color}Order size calculation failed or resulted in zero/negative size. No order placed for {trade_side.value} signal."
                            )

            # --- Graceful Shutdown ---
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down gracefully...")
                break  # Exit the main loop

            # --- Critical Error Handling (Stop the bot) ---
            except ccxt.AuthenticationError as e:
                logger.critical(
                    f"{Fore.RED}CRITICAL: Authentication failed during main loop: {e}. Check API keys/permissions. Exiting."
                )
                break  # Exit loop on critical auth error
            except ccxt.InvalidNonce as e:
                logger.critical(
                    f"{Fore.RED}CRITICAL: Invalid Nonce error: {e}. Synchronization issue with exchange API. Exiting."
                )
                break
            except ccxt.NotSupported as e:
                logger.critical(
                    f"{Fore.RED}CRITICAL: Operation not supported by exchange: {e}. Configuration or logic error. Exiting."
                )
                break

            # --- Potentially Recoverable Errors (Log and Continue) ---
            except ccxt.DDoSProtection as e:
                logger.error(
                    f"DDoS Protection or Rate Limit encountered: {e}. Loop will continue after delay."
                )
                # Decorator should handle retries, but loop delay will also help
            except ccxt.ExchangeNotAvailable as e:
                logger.error(
                    f"Exchange Not Available: {e}. Loop will continue, hoping for recovery."
                )
                # Consider adding logic to pause trading after repeated errors
            except ccxt.NetworkError as e:
                logger.error(
                    f"Network error occurred in main loop: {e}. Loop will continue."
                )
            except ccxt.ExchangeError as e:
                # Log other exchange errors but potentially continue if recoverable
                logger.error(
                    f"Exchange error occurred in main loop: {e}. Continuing..."
                )
                # Consider adding logic to halt trading temporarily if errors persist

            # --- Catch-all for Unexpected Errors ---
            except Exception:
                logger.exception("An unexpected error occurred in the main loop.")
                # Decide whether to continue or break on general errors
                # Consider adding a maximum consecutive error count before exiting
                # For safety, let's break on unexpected errors for now.
                logger.critical("Exiting due to unexpected error.")
                break  # Option to stop on any unexpected error

            # --- Loop Delay ---
            self._wait_for_next_iteration(start_time)

        logger.info("Scalping Bot main loop terminated.")

    def _wait_for_next_iteration(self, loop_start_time: float):
        """Calculates elapsed time and sleeps for the remaining loop delay."""
        end_time = time.monotonic()
        elapsed = end_time - loop_start_time
        sleep_time = max(0, self.trade_loop_delay - elapsed)
        if sleep_time > 0:
            logger.debug(
                f"Iteration took {elapsed:.3f}s. Sleeping for {sleep_time:.3f}s..."
            )
            time.sleep(sleep_time)
        else:
            logger.warning(
                f"Iteration took {elapsed:.3f}s, exceeding loop delay of {self.trade_loop_delay}s. Running next iteration immediately."
            )


# --- Main Execution ---
if __name__ == "__main__":
    exit_code = 0
    bot_instance: Optional[ScalpingBot] = None
    try:
        # Determine config file path (allow override from command line)
        config_arg = sys.argv[1] if len(sys.argv) > 1 else CONFIG_FILE_DEFAULT
        config_path = Path(config_arg)

        # Initialize the bot (loads config, sets up logger, connects to exchange)
        bot_instance = ScalpingBot(config_file=config_path)

        # Run the main trading loop
        bot_instance.run()

    # --- Specific Initialization Error Handling ---
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: Configuration file not found. {e}")
        exit_code = 1
    except (ConfigValidationError, yaml.YAMLError, ValueError) as e:
        # Catch config loading, parsing, validation, or market loading errors
        logger.error(
            f"Initialization failed due to invalid configuration or market setup: {e}"
        )
        exit_code = 1
    except (
        ccxt.AuthenticationError,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.NotSupported,
        AttributeError,
    ) as e:
        # Catch CCXT/exchange related errors during initialization
        logger.error(
            f"Initialization failed due to exchange connection or setup issue: {type(e).__name__}: {e}"
        )
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user. Exiting.")
        exit_code = 0  # Not an error exit
    except Exception as e:
        # Catch any other unexpected error during initialization
        logger.exception(
            f"An unexpected critical error occurred during initialization: {e}"
        )
        exit_code = 1

    # --- Runtime Error Handling (already handled in run(), but for completeness) ---
    # The run() method catches most runtime errors and decides whether to continue or break.
    # If run() exits due to a critical error caught within its loop, exit_code might still be 0 here.
    # We could potentially pass an exit code from run() if needed.

    finally:
        # --- Cleanup ---
        # (Add any resource cleanup needed here, e.g., closing connections if not handled by CCXT)
        logger.info(f"Exiting Scalping Bot with code {exit_code}.")
        logging.shutdown()  # Ensure all logs are flushed and handlers closed
        sys.exit(exit_code)
