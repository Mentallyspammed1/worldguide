# Enhanced Python Trading Bot Code
# Incorporates improvements based on the provided enhancement plan.

import ccxt
import pandas as pd
import json
import logging
import logging.handlers
import time
import sys
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum
from pathlib import Path

# --- Configuration ---

# Set Decimal precision (adjust as needed for your asset's precision)
DECIMAL_PRECISION = 8
getcontext().prec = DECIMAL_PRECISION + 4  # Add buffer for calculations
getcontext().rounding = ROUND_HALF_UP

CONFIG_FILE = Path("config.json")
STATE_FILE = Path("bot_state.json")
LOG_FILE = Path("trading_bot.log")
LOG_LEVEL = logging.INFO  # Default log level

# --- Enums ---


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


# --- Helper Functions ---


def setup_logger() -> logging.Logger:
    """Sets up the application logger with console and file handlers."""
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
    logger = logging.getLogger("TradingBot")
    logger.setLevel(LOG_LEVEL)

    # Prevent adding handlers multiple times if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    # File Handler (Rotating)
    try:
        log_dir = LOG_FILE.parent
        log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,  # 5 MB per file, 3 backups
        )
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to set up file logging handler: {e}", exc_info=True)

    # Suppress noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)  # Adjust if more ccxt detail needed

    return logger


def decimal_serializer(obj: Any) -> Union[str, Any]:
    """JSON serializer for Decimal objects."""
    if isinstance(obj, Decimal):
        # Ensure finite representation for JSON
        if obj.is_nan():
            return "NaN"
        if obj.is_infinite():
            return "Infinity" if obj > 0 else "-Infinity"
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def decimal_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    """JSON decoder hook to convert strings back to Decimal where appropriate."""
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                # Attempt conversion, but only if it looks like a number or special value
                if value.replace(".", "", 1).replace("-", "", 1).isdigit():
                    dct[key] = Decimal(value)
                elif value.lower() == "nan":
                    dct[key] = Decimal("NaN")
                elif value.lower() == "infinity":
                    dct[key] = Decimal("Infinity")
                elif value.lower() == "-infinity":
                    dct[key] = Decimal("-Infinity")
            except InvalidOperation:
                pass  # Keep as string if not a valid Decimal or special value
        elif isinstance(value, list):
            # Recursively decode lists
            dct[key] = [
                decimal_decoder(item)
                if isinstance(item, dict)
                else (
                    Decimal(item)
                    if isinstance(item, str) and item.replace(".", "", 1).replace("-", "", 1).isdigit()
                    else item
                )
                for item in value
            ]
        elif isinstance(value, dict):
            # Recursively decode nested dictionaries
            dct[key] = decimal_decoder(value)
    return dct


def load_config(config_path: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads configuration from a JSON file with Decimal conversion and validation."""
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return None
    try:
        with open(config_path, "r") as f:
            # Use object_pairs_hook for better Decimal handling in nested structures
            config = json.load(f, object_pairs_hook=lambda pairs: decimal_decoder(dict(pairs)))
        logger.info(f"Configuration loaded successfully from {config_path}")
        # --- Add specific config validation here ---
        if not validate_config(config, logger):
            logger.error("Configuration validation failed.")
            return None
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return None


def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validates the loaded configuration."""
    is_valid = True  # Assume valid initially
    required_keys = [
        "exchange",
        "api_credentials",
        "trading_settings",
        "indicator_settings",
        "risk_management",
        "logging",
    ]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration section: '{key}'")
            is_valid = False

    # If basic sections are missing, don't proceed with detailed checks
    if not is_valid:
        return False

    # Example: Validate specific parameters within sections
    exchange_cfg = config.get("exchange", {})
    if not isinstance(exchange_cfg.get("id"), str):
        logger.error("Config validation failed: 'exchange.id' must be a string.")
        is_valid = False

    api_creds = config.get("api_credentials", {})
    if not isinstance(api_creds.get("api_key"), str) or not api_creds.get("api_key"):
        logger.error("Config validation failed: 'api_credentials.api_key' must be a non-empty string.")
        is_valid = False
    if not isinstance(api_creds.get("api_secret"), str) or not api_creds.get("api_secret"):
        logger.error("Config validation failed: 'api_credentials.api_secret' must be a non-empty string.")
        is_valid = False

    settings = config.get("trading_settings", {})
    if not isinstance(settings.get("symbol"), str) or not settings.get("symbol"):
        logger.error("Config validation failed: 'trading_settings.symbol' must be a non-empty string.")
        is_valid = False
    if not isinstance(settings.get("timeframe"), str) or not settings.get("timeframe"):
        logger.error("Config validation failed: 'trading_settings.timeframe' must be a non-empty string.")
        is_valid = False
    leverage = settings.get("leverage")
    if not isinstance(leverage, Decimal) or leverage <= 0:
        logger.error(
            "Config validation failed: 'trading_settings.leverage' must be a positive number (loaded as Decimal)."
        )
        is_valid = False
    if not isinstance(settings.get("quote_asset"), str) or not settings.get("quote_asset"):
        logger.error("Config validation failed: 'trading_settings.quote_asset' must be a non-empty string.")
        is_valid = False

    indicators = config.get("indicator_settings", {})
    if "ema_short_period" in indicators and "ema_long_period" in indicators:
        if not (isinstance(indicators["ema_short_period"], int) and indicators["ema_short_period"] > 0):
            logger.error("Config validation failed: 'ema_short_period' must be a positive integer.")
            is_valid = False
        if not (isinstance(indicators["ema_long_period"], int) and indicators["ema_long_period"] > 0):
            logger.error("Config validation failed: 'ema_long_period' must be a positive integer.")
            is_valid = False
        if is_valid and indicators["ema_short_period"] >= indicators["ema_long_period"]:
            logger.error("Config validation failed: 'ema_short_period' must be less than 'ema_long_period'.")
            is_valid = False

    risk = config.get("risk_management", {})
    risk_percent = risk.get("risk_per_trade_percent")
    if not isinstance(risk_percent, Decimal) or not (Decimal(0) < risk_percent <= Decimal(100)):
        logger.error(
            "Config validation failed: 'risk_per_trade_percent' must be a number between 0 (exclusive) and 100 (inclusive)."
        )
        is_valid = False

    if is_valid:
        logger.info("Configuration validation successful.")
    return is_valid


def load_state(state_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Loads the bot's state from a JSON file with Decimal conversion."""
    if not state_path.is_file():
        logger.warning(f"State file not found at {state_path}. Starting with empty state.")
        return {}  # Return empty dict if no state file
    try:
        with open(state_path, "r") as f:
            # Use object_pairs_hook for better Decimal handling in nested structures
            state = json.load(f, object_pairs_hook=lambda pairs: decimal_decoder(dict(pairs)))
        logger.info(f"Bot state loaded successfully from {state_path}")
        return state
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON state file {state_path}: {e}. Using empty state.", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Error loading state from {state_path}: {e}. Using empty state.", exc_info=True)
        return {}


def save_state(state: Dict[str, Any], state_path: Path, logger: logging.Logger) -> None:
    """Saves the bot's state to a JSON file with Decimal serialization."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(state_path, "w") as f:
            json.dump(state, f, indent=4, default=decimal_serializer)
        logger.debug(f"Bot state saved successfully to {state_path}")
    except TypeError as e:
        logger.error(f"Error serializing state for saving: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving state to {state_path}: {e}", exc_info=True)


# --- Exchange Interaction Wrapper ---


class BybitV5Wrapper:
    """
    Wraps CCXT exchange interactions, focusing on Bybit V5 specifics,
    error handling, Decimal usage, and rate limiting.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config
        self.exchange_id = config["exchange"].get("id", "bybit")
        self.category = config["trading_settings"].get("category", "linear")  # linear, inverse, spot
        self.hedge_mode = config["trading_settings"].get("hedge_mode", False)  # Bybit hedge mode
        self.max_retries = config["exchange"].get("max_retries", 3)
        self.retry_delay = config["exchange"].get("retry_delay_seconds", 5)

        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": config["api_credentials"]["api_key"],
                "secret": config["api_credentials"]["api_secret"],
                "enableRateLimit": True,  # Let CCXT handle basic rate limiting
                "options": {
                    "defaultType": "swap" if self.category in ["linear", "inverse"] else "spot",
                    "adjustForTimeDifference": True,
                    "broker_id": config["exchange"].get("broker_id", None),  # Optional broker ID
                    # Add other necessary options
                },
            }
        )
        # Load markets to get precision details, etc.
        try:
            self.exchange.load_markets()
            self.logger.info(f"Markets loaded successfully for {self.exchange_id}.")
        except ccxt.AuthenticationError:
            self.logger.exception("Authentication failed. Check API keys.")
            raise  # Re-raise critical error
        except ccxt.NetworkError as e:
            self.logger.exception(f"Network error loading markets: {e}")
            raise  # Re-raise critical error
        except ccxt.ExchangeError as e:
            self.logger.exception(f"Exchange error loading markets: {e}")
            raise  # Re-raise critical error
        except Exception as e:
            self.logger.exception(f"Unexpected error loading markets: {e}")
            raise  # Re-raise critical error

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gets market data for a symbol."""
        try:
            market = self.exchange.market(symbol)
            if market:
                # Ensure precision values are loaded correctly
                if "precision" not in market or not market["precision"]:
                    self.logger.warning(f"Precision info missing for market {symbol}. Reloading markets.")
                    self.exchange.load_markets(reload=True)
                    market = self.exchange.market(symbol)
                    if "precision" not in market or not market["precision"]:
                        self.logger.error(f"Failed to load precision info for {symbol} even after reload.")
                        return None  # Cannot proceed without precision
            return market
        except ccxt.BadSymbol:
            self.logger.error(f"Symbol '{symbol}' not found on {self.exchange_id}.")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}", exc_info=True)
            return None

    def safe_ccxt_call(self, method_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Safely executes a CCXT method with retries and enhanced error handling.
        Distinguishes between retryable and non-retryable errors.
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                method = getattr(self.exchange, method_name)
                # Add category for V5 unified account calls if needed
                params = kwargs.get("params", {})
                if self.exchange_id == "bybit" and "category" not in params and self.category:
                    # Only add if method likely needs it and it's not already present
                    if method_name in [
                        "create_order",
                        "edit_order",
                        "cancel_order",
                        "fetch_positions",
                        "fetch_balance",
                        "set_leverage",
                        "set_margin_mode",
                        "fetch_open_orders",
                        "fetch_closed_orders",
                        "fetch_my_trades",
                        "private_post_position_trading_stop",
                    ]:  # Added specific private call
                        params["category"] = self.category
                        kwargs["params"] = params  # Update kwargs

                self.logger.debug(f"Calling CCXT method: {method_name}, Args: {args}, Kwargs: {kwargs}")
                result = method(*args, **kwargs)
                self.logger.debug(f"CCXT call {method_name} successful. Result snippet: {str(result)[:150]}...")
                return result

            # --- Specific CCXT/Bybit Error Handling ---
            except ccxt.AuthenticationError as e:
                self.logger.error(f"Authentication Error: {e}. Check API keys. Non-retryable.")
                return None  # Non-retryable
            except ccxt.PermissionDenied as e:
                self.logger.error(f"Permission Denied: {e}. Check API key permissions. Non-retryable.")
                return None  # Non-retryable
            except ccxt.AccountSuspended as e:
                self.logger.error(f"Account Suspended: {e}. Non-retryable.")
                return None  # Non-retryable
            except ccxt.InvalidOrder as e:
                self.logger.error(
                    f"Invalid Order parameters: {e}. Check order details (size, price, etc.). Non-retryable."
                )
                return None  # Non-retryable
            except ccxt.InsufficientFunds as e:
                self.logger.error(f"Insufficient Funds: {e}. Non-retryable.")
                return None  # Non-retryable
            except ccxt.BadSymbol as e:
                self.logger.error(f"Invalid Symbol: {e}. Non-retryable.")
                return None  # Non-retryable
            except ccxt.BadRequest as e:
                # Often parameter errors, potentially Bybit specific codes
                self.logger.error(f"Bad Request: {e}. Check parameters. May be non-retryable.")
                # Could parse e.args[0] for specific Bybit error codes if needed
                return None  # Assume non-retryable unless specific codes indicate otherwise
            except ccxt.RateLimitExceeded as e:
                self.logger.warning(
                    f"Rate Limit Exceeded: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})"
                )
                # CCXT's built-in rate limiter might handle this, but explicit retry adds robustness
            except ccxt.NetworkError as e:  # Includes ConnectionError, Timeout, etc.
                self.logger.warning(
                    f"Network Error: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})"
                )
            except ccxt.ExchangeNotAvailable as e:
                self.logger.warning(
                    f"Exchange Not Available: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})"
                )
            except ccxt.ExchangeError as e:  # General exchange error, potentially retryable
                # Check for specific Bybit codes that might be non-retryable (e.g., margin errors)
                msg = str(e).lower()
                if "margin" in msg or "position idx" in msg:  # Example non-retryable conditions
                    self.logger.error(f"Potentially non-retryable Exchange Error: {e}.")
                    return None
                self.logger.warning(
                    f"Generic Exchange Error: {e}. Retrying in {self.retry_delay}s... (Attempt {retries + 1}/{self.max_retries + 1})"
                )

            # --- Retry Logic ---
            retries += 1
            if retries <= self.max_retries:
                time.sleep(self.retry_delay)
            else:
                self.logger.error(f"CCXT call '{method_name}' failed after {self.max_retries + 1} attempts.")
                return None
        return None  # Should not be reached, but added for completeness

    # --- Specific API Call Wrappers ---

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data and returns it as a Pandas DataFrame with Decimal types."""
        self.logger.info(f"Fetching {limit} OHLCV candles for {symbol} ({timeframe})...")
        ohlcv = self.safe_ccxt_call("fetch_ohlcv", symbol, timeframe, limit=limit)
        if ohlcv is None or not ohlcv:
            self.logger.error(f"Failed to fetch OHLCV data for {symbol}.")
            return None

        try:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            # Convert OHLCV to Decimal AFTER initial DataFrame creation
            for col in ["open", "high", "low", "close", "volume"]:
                # Handle potential None values before conversion
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal("NaN"))
            df.set_index("timestamp", inplace=True)
            # Validate data sanity (e.g., check for NaNs, zero prices)
            if df[["open", "high", "low", "close"]].isnull().values.any():
                self.logger.warning(f"NaN values found in fetched OHLCV data for {symbol}.")
            if (df[["open", "high", "low", "close"]] <= Decimal(0)).values.any():
                self.logger.warning(f"Zero or negative values found in fetched OHLCV prices for {symbol}.")

            self.logger.info(f"Successfully fetched and processed {len(df)} OHLCV candles.")
            return df
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
            return None

    def fetch_balance(self) -> Optional[Dict[str, Decimal]]:
        """Fetches account balance, returning amounts as Decimals."""
        self.logger.debug("Fetching account balance...")
        # Bybit V5 requires category for fetch_balance
        params = {"category": self.category} if self.exchange_id == "bybit" else {}
        balance_data = self.safe_ccxt_call("fetch_balance", params=params)

        if balance_data is None:
            self.logger.error("Failed to fetch balance.")
            return None

        balances = {}
        try:
            # CCXT unified balance structure usually has 'free', 'used', 'total'
            # We primarily need the 'free' or 'available' balance for placing new trades,
            # and 'total' or 'equity' for risk calculation base.
            # Let's extract 'total' for risk calc and potentially 'free' if needed elsewhere.

            # Bybit V5 structure (example for UNIFIED account, may vary)
            if self.exchange_id == "bybit" and self.category in ["linear", "inverse"]:
                # Find the relevant account type in the response list
                account_type_map = {"linear": "CONTRACT", "inverse": "CONTRACT", "spot": "SPOT"}  # Adjust if needed
                target_account_type = account_type_map.get(self.category)

                account_list = balance_data.get("info", {}).get("result", {}).get("list", [])
                relevant_account = None
                if account_list:
                    # Prefer UNIFIED, then CONTRACT/SPOT based on category
                    relevant_account = next((acc for acc in account_list if acc.get("accountType") == "UNIFIED"), None)
                    if not relevant_account and target_account_type:
                        relevant_account = next(
                            (acc for acc in account_list if acc.get("accountType") == target_account_type), None
                        )

                if relevant_account:
                    coin_data = relevant_account.get("coin", [])
                    for coin_info in coin_data:
                        asset = coin_info.get("coin")
                        # Use 'equity' for total value including PnL, 'walletBalance' for cash balance
                        # 'availableToWithdraw' / 'availableToBorrow' might correspond to 'free'
                        total_balance_str = coin_info.get("equity")  # Use equity as the base for risk %
                        if asset and total_balance_str is not None:
                            try:
                                balances[asset] = Decimal(str(total_balance_str))
                            except InvalidOperation:
                                self.logger.warning(
                                    f"Could not convert balance for {asset} to Decimal: {total_balance_str}"
                                )
                else:
                    self.logger.warning(
                        f"Could not find relevant account type ('UNIFIED' or '{target_account_type}') in balance response for Bybit V5."
                    )
                    # Fallback to parsing the top-level 'total' if info structure fails
                    for asset, bal_info in balance_data.get("total", {}).items():
                        if bal_info is not None:
                            try:
                                balances[asset] = Decimal(str(bal_info))
                            except InvalidOperation:
                                self.logger.warning(
                                    f"Could not convert balance (fallback) for {asset} to Decimal: {bal_info}"
                                )

            else:  # Spot or other exchanges (use standard CCXT structure)
                for asset, bal_info in balance_data.get("total", {}).items():
                    if bal_info is not None:
                        try:
                            balances[asset] = Decimal(str(bal_info))
                        except InvalidOperation:
                            self.logger.warning(f"Could not convert balance for {asset} to Decimal: {bal_info}")

            if not balances:
                self.logger.warning("Parsed balance data is empty.")
                self.logger.debug(f"Raw balance data: {balance_data}")
            else:
                self.logger.info(f"Balance fetched successfully. Assets: {list(balances.keys())}")
            return balances

        except Exception as e:
            self.logger.error(f"Error parsing balance data: {e}", exc_info=True)
            self.logger.debug(f"Raw balance data: {balance_data}")
            return None

    def fetch_positions(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Fetches open positions, converting relevant fields to Decimal."""
        self.logger.debug(f"Fetching positions for symbol: {symbol or 'all'}...")
        params = {"category": self.category} if self.exchange_id == "bybit" else {}
        symbols_arg = [symbol] if symbol else None  # Pass symbol in list for unified fetch_positions

        # Bybit V5 specific: Can filter by symbol directly in params
        if symbol and self.exchange_id == "bybit":
            params["symbol"] = symbol
            symbols_arg = None  # Don't pass symbols list if using params filter

        positions_data = self.safe_ccxt_call("fetch_positions", symbols=symbols_arg, params=params)

        if positions_data is None:
            self.logger.error(f"Failed to fetch positions for {symbol or 'all'}.")
            return None

        processed_positions = []
        try:
            for pos in positions_data:
                # --- Reliable check for active position ---
                # Use 'size' from 'info' for Bybit V5 as primary check
                size_str = pos.get("info", {}).get("size", "0")
                try:
                    size_dec = Decimal(size_str)
                    if size_dec == Decimal(0):
                        continue  # Skip zero-size positions explicitly
                except InvalidOperation:
                    self.logger.warning(f"Could not parse position size '{size_str}' as Decimal. Skipping position.")
                    continue

                processed = pos.copy()  # Work on a copy

                # --- Convert relevant fields to Decimal ---
                # Standard CCXT fields
                decimal_fields_std = [
                    "contracts",
                    "contractSize",
                    "entryPrice",
                    "leverage",
                    "liquidationPrice",
                    "markPrice",
                    "notional",
                    "unrealizedPnl",
                    "initialMargin",
                    "maintenanceMargin",
                    "initialMarginPercentage",
                    "maintenanceMarginPercentage",
                ]
                # Bybit V5 specific fields often in 'info'
                decimal_fields_info = [
                    "avgPrice",
                    "cumRealisedPnl",
                    "liqPrice",
                    "markPrice",
                    "positionValue",
                    "stopLoss",
                    "takeProfit",
                    "trailingStop",
                    "unrealisedPnl",
                    "size",
                    "positionIM",
                    "positionMM",
                ]

                for field in decimal_fields_std:
                    if field in processed and processed[field] is not None:
                        try:
                            processed[field] = Decimal(str(processed[field]))
                        except InvalidOperation:
                            self.logger.warning(
                                f"Could not convert standard position field '{field}' to Decimal: {processed[field]}"
                            )
                            processed[field] = Decimal("NaN")  # Mark as NaN if conversion fails

                if "info" in processed and isinstance(processed["info"], dict):
                    info = processed["info"]
                    for field in decimal_fields_info:
                        if field in info and info[field] is not None and info[field] != "":
                            try:
                                info[field] = Decimal(str(info[field]))
                            except InvalidOperation:
                                self.logger.warning(
                                    f"Could not convert position info field '{field}' to Decimal: {info[field]}"
                                )
                                info[field] = Decimal("NaN")  # Mark as NaN

                # --- Determine position side more reliably ---
                # Use Bybit V5 'side' field from 'info' first
                side = None
                if "info" in processed and "side" in processed["info"]:
                    side_str = processed["info"]["side"].lower()
                    if side_str == "buy":
                        side = "long"
                    elif side_str == "sell":
                        side = "short"
                    elif side_str == "none":  # Explicit 'None' side from Bybit
                        side = "none"

                # Fallback to CCXT 'side' if info side is missing/ambiguous
                if side is None and "side" in processed and processed["side"]:
                    side = processed["side"].lower()

                # Final check: if side is still None or 'none', but size > 0, infer from size sign if possible (though Bybit V5 size is unsigned)
                # Relying on the explicit 'side' field is safer. If side is 'none' but size > 0, log a warning.
                if side == "none" and size_dec != Decimal(0):
                    self.logger.warning(
                        f"Position for {processed.get('symbol')} has size {size_dec} but side is 'None'. Check exchange data."
                    )
                    # Cannot reliably determine side, might need manual intervention or different logic. Skip?
                    # continue

                processed["side"] = side  # Update the main 'side' field

                # Add positionIdx for hedge mode if available
                if "info" in processed and "positionIdx" in processed["info"]:
                    processed["positionIdx"] = int(processed["info"]["positionIdx"])  # Store as int

                processed_positions.append(processed)

            self.logger.info(f"Fetched {len(processed_positions)} active position(s).")
            return processed_positions
        except Exception as e:
            self.logger.error(f"Error processing position data: {e}", exc_info=True)
            self.logger.debug(f"Raw positions data: {positions_data}")
            return None

    def format_value_for_api(self, symbol: str, value_type: str, value: Decimal) -> Union[str, float]:
        """Formats amount or price to string based on market precision for API calls."""
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f"Invalid Decimal value for formatting: {value}")

        market = self.get_market(symbol)
        if not market:
            raise ValueError(f"Market data not found for {symbol}, cannot format value.")

        if value_type == "amount":
            # Use amount_to_precision for formatting quantity
            formatted_value = self.exchange.amount_to_precision(symbol, float(value))
        elif value_type == "price":
            # Use price_to_precision for formatting price
            formatted_value = self.exchange.price_to_precision(symbol, float(value))
        else:
            raise ValueError(f"Invalid value_type: {value_type}. Use 'amount' or 'price'.")

        # CCXT formatting methods return strings. Return as string for consistency.
        # Some APIs might strictly require floats, but sending precise strings is generally safer.
        return formatted_value

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: OrderSide,
        amount: Decimal,
        price: Optional[Decimal] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Creates an order, ensuring amount and price are correctly formatted using market precision."""
        self.logger.info(
            f"Attempting to create {side.value} {order_type} order for {amount} {symbol} @ {price or 'market'}..."
        )

        market = self.get_market(symbol)
        if not market:
            self.logger.error(f"Cannot create order: Market data for {symbol} not found.")
            return None
        if amount <= 0:
            self.logger.error(f"Cannot create order: Amount must be positive ({amount}).")
            return None

        try:
            # Format amount and price using market precision rules via helper function
            amount_str = self.format_value_for_api(symbol, "amount", amount)
            price_str = (
                self.format_value_for_api(symbol, "price", price)
                if price is not None and order_type != "market"
                else None
            )

            self.logger.debug(f"Formatted order values: Amount='{amount_str}', Price='{price_str}'")

        except ValueError as e:
            self.logger.error(f"Error formatting order values: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during value formatting: {e}", exc_info=True)
            return None

        # --- Bybit V5 Specific Parameters ---
        order_params = {
            # 'category': self.category, # safe_ccxt_call adds this
        }
        if self.hedge_mode:
            # Determine positionIdx based on side for hedge mode entries/exits
            # For opening: 1 for Buy/Long, 2 for Sell/Short
            # For closing: Needs to match the position being closed (handled in close_position)
            # This assumes create_order is used for opening here.
            order_params["positionIdx"] = 1 if side == OrderSide.BUY else 2

        # Combine with user-provided params (user params take precedence)
        if params:
            order_params.update(params)

        # Prepare arguments for safe_ccxt_call
        # Pass formatted strings directly, CCXT handles conversion if needed
        call_args = [symbol, order_type, side.value, amount_str]
        if price_str:
            call_args.append(price_str)
        else:
            # For market orders, price argument might be required as None or omitted depending on CCXT version/exchange
            # Passing None explicitly is safer.
            call_args.append(None)

        order_result = self.safe_ccxt_call("create_order", *call_args, params=order_params)

        if order_result:
            order_id = order_result.get("id")
            self.logger.info(f"Order creation request sent successfully. Order ID: {order_id}")
            # Further processing/parsing of the result can be done here if needed
            # e.g., converting fields back to Decimal
        else:
            self.logger.error(f"Failed to create {side.value} {order_type} order for {symbol}.")

        return order_result  # Return raw result

    def set_leverage(self, symbol: str, leverage: Decimal) -> bool:
        """Sets leverage for a symbol (Bybit V5 requires buy/sell leverage)."""
        self.logger.info(f"Setting leverage for {symbol} to {leverage}x...")
        if not (isinstance(leverage, Decimal) and leverage > 0 and leverage.is_finite()):
            self.logger.error(f"Invalid leverage value: {leverage}. Must be a positive finite Decimal.")
            return False

        leverage_str = str(leverage)  # Pass leverage as string for precision

        # Bybit V5 requires setting buyLeverage and sellLeverage separately
        # And requires the symbol argument
        params = {
            # 'category': self.category, # safe_ccxt_call adds this
            "buyLeverage": leverage_str,
            "sellLeverage": leverage_str,
        }
        # CCXT's set_leverage expects leverage (float/int), symbol, params
        # Pass leverage as float derived from Decimal for the main argument
        result = self.safe_ccxt_call("set_leverage", float(leverage), symbol, params=params)

        # Check result (structure might vary, check Bybit docs/ccxt implementation)
        # Bybit V5 setLeverage usually returns None on success and throws error on failure.
        # safe_ccxt_call returns the result or None on error.
        if result is not None:
            # Success might be indicated by a non-None result (even empty dict) or lack of error
            self.logger.info(f"Leverage for {symbol} set to {leverage}x successfully (or request accepted).")
            # Bybit might return specific codes in info, e.g., {'retCode': 0, ...}
            if isinstance(result, dict) and result.get("info", {}).get("retCode") == 0:
                self.logger.info("Bybit API confirmed successful leverage setting (retCode 0).")
                return True
            elif isinstance(result, dict) and not result.get("info"):
                self.logger.warning("set_leverage call returned non-empty dict but no 'info'. Assuming success.")
                return True
            elif result is None:  # Can happen if ccxt method returns None on success
                self.logger.info("set_leverage call returned None. Assuming success based on lack of error.")
                return True
            else:  # Unexpected result structure
                self.logger.warning(
                    f"set_leverage call returned unexpected result: {result}. Assuming success based on lack of error."
                )
                return True

        else:
            # Error handled and logged by safe_ccxt_call
            self.logger.error(f"Failed to set leverage for {symbol} to {leverage}x.")
            return False

    def set_protection(
        self,
        symbol: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        trailing_stop: Optional[Dict[str, Decimal]] = None,
        position_idx: Optional[int] = None,
    ) -> bool:
        """Sets stop loss, take profit, or trailing stop for a position using Bybit V5's trading stop endpoint."""
        action = []
        if stop_loss:
            action.append(f"SL={stop_loss}")
        if take_profit:
            action.append(f"TP={take_profit}")
        if trailing_stop:
            action.append(f"TSL={trailing_stop}")
        if not action:
            self.logger.warning("set_protection called with no protection levels specified.")
            return False

        self.logger.info(f"Attempting to set protection for {symbol}: {' / '.join(action)}")

        market = self.get_market(symbol)
        if not market:
            self.logger.error(f"Cannot set protection: Market data for {symbol} not found.")
            return False

        # --- Prepare Parameters for private_post_position_trading_stop ---
        params = {
            # 'category': self.category, # safe_ccxt_call adds this
            "symbol": symbol,
            # Bybit expects prices as strings
        }

        try:
            if stop_loss:
                if stop_loss <= 0:
                    raise ValueError("Stop loss must be positive.")
                params["stopLoss"] = self.format_value_for_api(symbol, "price", stop_loss)
            if take_profit:
                if take_profit <= 0:
                    raise ValueError("Take profit must be positive.")
                params["takeProfit"] = self.format_value_for_api(symbol, "price", take_profit)

            # Trailing Stop Handling (Bybit V5 specific)
            if trailing_stop:
                # Bybit uses 'trailingStop' for the distance/value (price or percentage points)
                ts_value = trailing_stop.get("distance") or trailing_stop.get("value")
                ts_active_price = trailing_stop.get("activation_price")

                if ts_value is not None:
                    if ts_value <= 0:
                        raise ValueError("Trailing stop distance/value must be positive.")
                    # Assuming ts_value is a price distance here. If percentage, formatting might differ.
                    # Bybit API expects this as a string value representing the trail distance.
                    # Formatting as 'price' might work if it's a price value, but needs verification.
                    # Let's assume it's a price delta and format it like a price for now.
                    # WARNING: Verify if Bybit expects TSL distance formatted as price or amount/other.
                    # For safety, let's just convert to string directly for TSL value.
                    params["trailingStop"] = str(ts_value)  # Pass TSL distance/value as string
                    # params['tpslMode'] = 'Partial' # If needed, e.g., for partial TP/SL
                    # params['tpTriggerBy'] = 'MarkPrice' # Or LastPrice, IndexPrice
                    # params['slTriggerBy'] = 'MarkPrice' # Or LastPrice, IndexPrice

                if ts_active_price is not None:
                    if ts_active_price <= 0:
                        raise ValueError("Trailing stop activation price must be positive.")
                    params["activePrice"] = self.format_value_for_api(symbol, "price", ts_active_price)

        except ValueError as e:
            self.logger.error(f"Invalid protection parameter value: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error formatting protection parameters: {e}", exc_info=True)
            return False

        # Position Index for Hedge Mode
        # Required by Bybit V5 for hedge mode positions
        if self.hedge_mode:
            if position_idx is None:
                self.logger.error(
                    "Hedge mode active, but positionIdx is required for set_protection and was not provided."
                )
                return False
            params["positionIdx"] = position_idx
        elif "positionIdx" in params:
            # Remove if not in hedge mode, might cause errors otherwise
            del params["positionIdx"]

        # --- Execute API Call ---
        # Use the specific private method for Bybit V5 trading stop
        result = self.safe_ccxt_call("private_post_position_trading_stop", params=params)

        # --- Process Result ---
        if result is not None:
            # Check Bybit V5 API docs for response structure on success/failure
            ret_code = result.get("retCode")
            ret_msg = result.get("retMsg", "No message")
            ext_info = result.get("retExtInfo", {})

            if ret_code == 0:
                self.logger.info(
                    f"Protection levels set successfully for {symbol} (positionIdx: {position_idx if self.hedge_mode else 'N/A'})."
                )
                return True
            else:
                # Log specific error code/message
                self.logger.error(
                    f"Failed to set protection for {symbol}. Code: {ret_code}, Msg: {ret_msg}, Extra: {ext_info}"
                )
                self.logger.debug(f"Params sent: {params}")
                return False
        else:
            # safe_ccxt_call already logged the failure reason (e.g., network error, auth error)
            self.logger.error(f"API call failed for setting protection on {symbol}.")
            return False


# --- Trading Strategy Analyzer ---


class TradingAnalyzer:
    """Analyzes market data using technical indicators to generate trading signals."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.config = config.get("indicator_settings", {})
        self.weights = self.config.get("signal_weights", {})
        if not self.weights:
            self.logger.warning("No 'signal_weights' found in indicator_settings. Using default weights.")
            # Ensure default weights are Decimals
            self.weights = {
                "rsi": Decimal("0.3"),
                "macd": Decimal("0.4"),
                "ema_cross": Decimal("0.3"),
            }
        else:
            # Ensure loaded weights are Decimals
            self.weights = {k: Decimal(str(v)) for k, v in self.weights.items()}

        # Normalize weights if they don't sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - Decimal("1.0")) > Decimal("1e-9"):  # Use tolerance for float conversion issues
            self.logger.warning(f"Signal weights sum to {total_weight}, not 1. Normalizing.")
            if total_weight == Decimal(0):
                self.logger.error("Total signal weight is zero, cannot normalize. Disabling weighted signals.")
                self.weights = {}  # Or handle differently
            else:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all configured technical indicators."""
        if ohlcv_df is None or ohlcv_df.empty:
            self.logger.error("Cannot calculate indicators: OHLCV data is missing or empty.")
            return None
        # Ensure required columns exist and are Decimals
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in ohlcv_df.columns for col in required_cols):
            self.logger.error(f"Missing required OHLCV columns in DataFrame: {required_cols}")
            return None
        if not all(isinstance(ohlcv_df[col].iloc[0], Decimal) for col in required_cols):
            self.logger.error("OHLCV columns must be Decimal type for indicator calculation.")
            return None

        self.logger.debug(f"Calculating indicators for {len(ohlcv_df)} candles...")
        df = ohlcv_df.copy()

        # Convert Decimal columns to float for pandas_ta, handle potential NaNs/Infs
        float_cols = ["open", "high", "low", "close", "volume"]
        for col in float_cols:
            try:
                # Replace non-finite Decimals before converting to float
                df[col] = df[col].apply(lambda x: float(x) if x.is_finite() else pd.NA)
                df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, coercing errors
            except Exception as e:
                self.logger.error(f"Error converting column {col} to float for TA calculation: {e}", exc_info=True)
                return None  # Cannot proceed if conversion fails

        # Drop rows with NaN in OHLC columns after conversion, as TA libs often fail
        initial_len = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            self.logger.warning(f"Dropped {initial_len - len(df)} rows with NaN in OHLC columns before TA calculation.")

        if df.empty:
            self.logger.error("DataFrame is empty after handling NaNs in OHLC data. Cannot calculate indicators.")
            return None

        try:
            # --- Calculate Indicators using pandas_ta ---
            # Use integer periods directly from config
            rsi_period = self.config.get("rsi_period")
            if isinstance(rsi_period, int) and rsi_period > 0:
                df.ta.rsi(length=rsi_period, append=True)

            macd_fast = self.config.get("macd_fast")
            macd_slow = self.config.get("macd_slow")
            macd_signal = self.config.get("macd_signal")
            if all(isinstance(p, int) and p > 0 for p in [macd_fast, macd_slow, macd_signal]):
                df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)

            ema_short_period = self.config.get("ema_short_period")
            if isinstance(ema_short_period, int) and ema_short_period > 0:
                df.ta.ema(length=ema_short_period, append=True)

            ema_long_period = self.config.get("ema_long_period")
            if isinstance(ema_long_period, int) and ema_long_period > 0:
                df.ta.ema(length=ema_long_period, append=True)

            atr_period = self.config.get("atr_period")
            if isinstance(atr_period, int) and atr_period > 0:
                df.ta.atr(length=atr_period, append=True, mamode="ema")  # Use EMA for ATR smoothing typically

            # Add other indicators as needed...

            self.logger.debug(f"Indicators calculated. Columns: {df.columns.tolist()}")

            # --- Convert calculated indicator columns back to Decimal ---
            # Identify newly added columns by pandas_ta
            original_cols = set(ohlcv_df.columns)
            new_cols = [col for col in df.columns if col not in original_cols]
            for col in new_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Convert float indicators back to Decimal for internal consistency
                    # Handle potential NaN/Inf from calculations
                    df[col] = df[col].apply(
                        lambda x: Decimal(str(x))
                        if pd.notna(x) and pd.api.types.is_number(x) and math.isfinite(x)
                        else Decimal("NaN")
                    )

            # Restore original Decimal types for OHLCV columns by merging back
            # This ensures we keep the original precision and Decimal type
            for col in float_cols:
                if col in ohlcv_df.columns:
                    df[col] = ohlcv_df[col]

            # Reindex df to match the original ohlcv_df index to include rows dropped earlier
            df = df.reindex(ohlcv_df.index)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
            return None

    def generate_signal(self, indicators_df: pd.DataFrame) -> Tuple[Signal, Dict[str, Any]]:
        """
        Generates a trading signal based on the latest indicator values
        and configured weighted scoring.
        Returns the final signal and contributing factors/scores.
        """
        if indicators_df is None or indicators_df.empty:
            self.logger.warning("Cannot generate signal: Indicators data is missing.")
            return Signal.HOLD, {}

        # Ensure index is datetime (or handle non-datetime index)
        if not isinstance(indicators_df.index, pd.DatetimeIndex):
            self.logger.warning("Indicators DataFrame index is not DatetimeIndex. Using iloc[-1].")
            try:
                latest_data = indicators_df.iloc[-1]
            except IndexError:
                self.logger.error("Cannot get latest data: DataFrame is empty.")
                return Signal.HOLD, {}
        else:
            # Get the absolute latest row based on time
            latest_data = indicators_df.iloc[-1]

        scores = {}
        contributing_factors = {}

        # --- Evaluate Individual Indicators ---
        # Ensure values are valid Decimals before comparison

        # RSI Example
        rsi_weight = self.weights.get("rsi", Decimal(0))
        rsi_period = self.config.get("rsi_period")
        rsi_col = f"RSI_{rsi_period}" if isinstance(rsi_period, int) else None
        if rsi_weight > 0 and rsi_col and rsi_col in latest_data:
            rsi_value = latest_data[rsi_col]
            if isinstance(rsi_value, Decimal) and rsi_value.is_finite():
                overbought = self.config.get("rsi_overbought", Decimal("70"))
                oversold = self.config.get("rsi_oversold", Decimal("30"))
                rsi_score = Decimal(0)
                if rsi_value > overbought:
                    rsi_score = Decimal("-1")  # Sell signal
                elif rsi_value < oversold:
                    rsi_score = Decimal("1")  # Buy signal
                scores["rsi"] = rsi_score * rsi_weight
                contributing_factors["rsi"] = {"value": rsi_value, "score": rsi_score, "weight": rsi_weight}
            else:
                self.logger.warning(f"Invalid RSI value for signal generation: {rsi_value}")

        # MACD Example (Histogram sign)
        macd_weight = self.weights.get("macd", Decimal(0))
        macd_fast = self.config.get("macd_fast", 12)
        macd_slow = self.config.get("macd_slow", 26)
        macd_signal_p = self.config.get("macd_signal", 9)
        macdh_col = (
            f"MACDh_{macd_fast}_{macd_slow}_{macd_signal_p}"
            if all(isinstance(p, int) for p in [macd_fast, macd_slow, macd_signal_p])
            else None
        )

        if macd_weight > 0 and macdh_col and macdh_col in latest_data:
            macd_hist = latest_data[macdh_col]
            if isinstance(macd_hist, Decimal) and macd_hist.is_finite():
                # Simple histogram sign check for cross direction
                macd_score = Decimal(0)
                # Add threshold to avoid noise near zero cross?
                hist_threshold = Decimal(str(self.config.get("macd_hist_threshold", "0")))
                if macd_hist > hist_threshold:  # MACD line crossed above signal line
                    macd_score = Decimal("1")  # Buy signal
                elif macd_hist < -hist_threshold:  # MACD line crossed below signal line
                    macd_score = Decimal("-1")  # Sell signal
                scores["macd"] = macd_score * macd_weight
                contributing_factors["macd"] = {"histogram": macd_hist, "score": macd_score, "weight": macd_weight}
            else:
                self.logger.warning(f"Invalid MACD Histogram value for signal generation: {macd_hist}")

        # EMA Cross Example
        ema_cross_weight = self.weights.get("ema_cross", Decimal(0))
        ema_short_period = self.config.get("ema_short_period")
        ema_long_period = self.config.get("ema_long_period")
        ema_short_col = f"EMA_{ema_short_period}" if isinstance(ema_short_period, int) else None
        ema_long_col = f"EMA_{ema_long_period}" if isinstance(ema_long_period, int) else None

        if (
            ema_cross_weight > 0
            and ema_short_col
            and ema_long_col
            and all(c in latest_data for c in [ema_short_col, ema_long_col])
        ):
            ema_short = latest_data[ema_short_col]
            ema_long = latest_data[ema_long_col]
            if (
                isinstance(ema_short, Decimal)
                and ema_short.is_finite()
                and isinstance(ema_long, Decimal)
                and ema_long.is_finite()
            ):
                ema_cross_score = Decimal(0)
                # Check for actual cross (previous state vs current state) might be more robust
                # Simple check: short > long is bullish
                if ema_short > ema_long:
                    ema_cross_score = Decimal("1")  # Bullish cross
                elif ema_short < ema_long:
                    ema_cross_score = Decimal("-1")  # Bearish cross
                scores["ema_cross"] = ema_cross_score * ema_cross_weight
                contributing_factors["ema_cross"] = {
                    "short_ema": ema_short,
                    "long_ema": ema_long,
                    "score": ema_cross_score,
                    "weight": ema_cross_weight,
                }
            else:
                self.logger.warning(f"Invalid EMA values for signal generation: Short={ema_short}, Long={ema_long}")

        # --- Combine Scores ---
        if not scores:
            self.logger.warning("No valid indicator scores generated. Defaulting to HOLD.")
            return Signal.HOLD, {"final_score": Decimal(0), "factors": contributing_factors}

        final_score = sum(scores.values())

        # --- Determine Final Signal ---
        # Ensure thresholds are Decimals
        strong_buy_threshold = self.config.get("strong_buy_threshold", Decimal("0.7"))
        buy_threshold = self.config.get("buy_threshold", Decimal("0.2"))
        sell_threshold = self.config.get("sell_threshold", Decimal("-0.2"))
        strong_sell_threshold = self.config.get("strong_sell_threshold", Decimal("-0.7"))

        signal = Signal.HOLD
        if final_score >= strong_buy_threshold:
            signal = Signal.STRONG_BUY
        elif final_score >= buy_threshold:
            signal = Signal.BUY
        elif final_score <= strong_sell_threshold:
            signal = Signal.STRONG_SELL
        elif final_score <= sell_threshold:
            signal = Signal.SELL

        self.logger.info(f"Signal generated: {signal.name} (Score: {final_score:.4f})")
        self.logger.debug(f"Contributing factors: {contributing_factors}")

        return signal, {"final_score": final_score, "factors": contributing_factors}


# --- Position and Risk Management ---
import math  # Import math for isfinite check


class PositionManager:
    """Handles position sizing, stop-loss, take-profit, and exit logic."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, exchange_wrapper: BybitV5Wrapper):
        self.logger = logger
        self.config = config
        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading_settings", {})
        self.indicator_config = config.get("indicator_settings", {})  # Added for convenience
        self.exchange = exchange_wrapper
        self.symbol = self.trading_config.get("symbol")
        self.hedge_mode = self.trading_config.get("hedge_mode", False)

    def get_base_quote(self) -> Tuple[Optional[str], Optional[str]]:
        """Gets base and quote assets from the symbol."""
        market = self.exchange.get_market(self.symbol)
        if market:
            return market.get("base"), market.get("quote")
        return None, None

    def calculate_position_size(
        self, entry_price: Decimal, stop_loss_price: Decimal, available_equity: Decimal, quote_asset: str
    ) -> Optional[Decimal]:
        """
        Calculates position size based on risk percentage of equity, stop distance.
        Returns size in base currency (e.g., BTC amount for BTC/USDT).
        """
        risk_percent = self.risk_config.get("risk_per_trade_percent", Decimal("1.0")) / Decimal("100")
        # Leverage is applied by the exchange, not directly in size calculation based on risk % of equity.
        # leverage = self.trading_config.get("leverage", Decimal("1"))

        if not all(
            isinstance(val, Decimal) and val.is_finite() and val > 0
            for val in [entry_price, stop_loss_price, available_equity]
        ):
            self.logger.error(
                f"Invalid inputs for position size calculation: entry={entry_price}, sl={stop_loss_price}, equity={available_equity}. Must be positive finite Decimals."
            )
            return None
        if entry_price == stop_loss_price:
            self.logger.error("Entry price and stop-loss price cannot be the same.")
            return None

        market = self.exchange.get_market(self.symbol)
        if not market:
            self.logger.error(f"Cannot calculate position size: Market data for {self.symbol} not found.")
            return None
        base_asset, market_quote_asset = self.get_base_quote()
        if not base_asset or not market_quote_asset:
            self.logger.error(f"Could not determine base/quote asset for {self.symbol}.")
            return None
        if market_quote_asset != quote_asset:
            self.logger.warning(
                f"Configured quote_asset '{quote_asset}' differs from market quote '{market_quote_asset}'. Using market quote '{market_quote_asset}' for calculations."
            )
            # This might indicate a config error, but proceed with market quote for now.

        # Amount of quote currency to risk based on total equity
        risk_amount_quote = available_equity * risk_percent
        self.logger.info(
            f"Risk per trade: {risk_percent:.2%}, Total Equity ({quote_asset}): {available_equity:.{DECIMAL_PRECISION}f}, Risk Amount: {risk_amount_quote:.{DECIMAL_PRECISION}f} {quote_asset}"
        )

        # Price distance for stop loss
        stop_loss_distance = abs(entry_price - stop_loss_price)
        if stop_loss_distance <= Decimal(0):
            self.logger.error(f"Stop loss distance is zero or negative ({stop_loss_distance}). Check SL price.")
            return None

        # --- Calculate position size in base currency ---
        # Formula depends on contract type (linear vs inverse)
        position_size_base = None
        if market.get("inverse", False):
            # Inverse: Size = (Risk Amount in Quote * Entry Price) / Stop Distance
            # Note: Risk is on the value of the position in quote currency.
            # PnL = Size_Base * (1/Exit - 1/Entry) => Risk = Size_Base * abs(1/SL - 1/Entry)
            # Risk_Quote = Risk_Base * SL_Price = Size_Base * abs(1/SL - 1/Entry) * SL_Price
            # Size_Base = Risk_Quote / (abs(1/SL - 1/Entry) * SL_Price)
            # Size_Base = Risk_Quote / (abs(Entry - SL)/(SL*Entry) * SL_Price)
            # Size_Base = Risk_Quote / (Stop_Distance / Entry) = Risk_Quote * Entry / Stop_Distance
            position_size_base = (risk_amount_quote * entry_price) / stop_loss_distance
            self.logger.debug("Calculating size for INVERSE contract.")
        else:  # Linear or Spot
            # Linear: Size = (Risk Amount in Quote) / (Stop Loss Distance in Quote per Base Unit)
            position_size_base = risk_amount_quote / stop_loss_distance
            self.logger.debug("Calculating size for LINEAR/SPOT contract.")

        if position_size_base is None or not position_size_base.is_finite() or position_size_base <= 0:
            self.logger.error(f"Calculated position size is invalid: {position_size_base}")
            return None

        self.logger.debug(f"Calculated raw position size: {position_size_base:.{DECIMAL_PRECISION}f} {base_asset}")

        # --- Apply Precision, Lot Size, Min/Max Limits ---
        try:
            # Get amount precision (number of decimal places for base currency)
            market.get("precision", {}).get("amount")
            # Get min/max order size limits
            min_amount = market.get("limits", {}).get("amount", {}).get("min")
            max_amount = market.get("limits", {}).get("amount", {}).get("max")

            # Convert limits to Decimal if they exist
            min_amount_dec = Decimal(str(min_amount)) if min_amount is not None else None
            max_amount_dec = Decimal(str(max_amount)) if max_amount is not None else None

            # 1. Check against minimum amount
            if min_amount_dec is not None and position_size_base < min_amount_dec:
                self.logger.warning(
                    f"Calculated position size {position_size_base:.{DECIMAL_PRECISION}f} {base_asset} is below minimum order size {min_amount_dec} {base_asset}. Cannot open position."
                )
                return None

            # 2. Apply amount precision (rounding/truncation)
            # Use format_value_for_api which uses exchange's precision rules
            formatted_size_str = self.exchange.format_value_for_api(self.symbol, "amount", position_size_base)
            final_size_base = Decimal(formatted_size_str)
            self.logger.debug(f"Position size after applying precision: {final_size_base} {base_asset}")

            # 3. Re-check minimum after precision adjustment
            if min_amount_dec is not None and final_size_base < min_amount_dec:
                self.logger.warning(
                    f"Position size {final_size_base} after precision is below minimum {min_amount_dec}. Cannot open position."
                )
                return None

            # 4. Check against maximum amount
            if max_amount_dec is not None and final_size_base > max_amount_dec:
                self.logger.warning(
                    f"Calculated position size {final_size_base} exceeds max limit {max_amount_dec}. Capping size to max limit."
                )
                final_size_base = max_amount_dec
                # Re-apply precision to the capped value
                formatted_size_str = self.exchange.format_value_for_api(self.symbol, "amount", final_size_base)
                final_size_base = Decimal(formatted_size_str)
                self.logger.info(f"Position size capped to: {final_size_base} {base_asset}")

            if final_size_base <= 0:
                self.logger.error("Final position size is zero or negative after adjustments. Cannot open position.")
                return None

            self.logger.info(f"Calculated Final Position Size: {final_size_base} {base_asset}")
            return final_size_base

        except ValueError as e:
            self.logger.error(f"Error applying market limits/precision to position size: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during position size finalization: {e}", exc_info=True)
            return None

    def quantize_price(self, price: Decimal, side: Optional[PositionSide] = None) -> Optional[Decimal]:
        """Quantizes price according to market precision rules, rounding conservatively for SL/TP."""
        if not isinstance(price, Decimal) or not price.is_finite():
            self.logger.error(f"Invalid price for quantization: {price}")
            return None

        market = self.exchange.get_market(self.symbol)
        if not market or "precision" not in market or "price" not in market["precision"]:
            self.logger.error(f"Cannot quantize price: Market or price precision not found for {self.symbol}.")
            # Fallback: return unquantized price with warning? Or fail? Let's fail.
            return None

        try:
            # Use the exchange's price_to_precision method first
            price_str = self.exchange.price_to_precision(self.symbol, float(price))
            quantized_price = Decimal(price_str)

            # Optional: Apply conservative rounding specifically for SL/TP after basic quantization
            # This might be redundant if price_to_precision already handles rounding correctly,
            # but adds an extra layer of safety. Requires understanding tick size.
            tick_size_str = market.get("precision", {}).get("price")
            if tick_size_str is not None:
                tick_size = Decimal(str(tick_size_str))
                if tick_size > 0:  # Ensure tick_size is valid
                    if side == PositionSide.LONG and price == self.state.get("stop_loss_price"):  # Round down Long SL
                        quantized_price = (price // tick_size) * tick_size
                    elif side == PositionSide.SHORT and price == self.state.get("stop_loss_price"):  # Round up Short SL
                        quantized_price = math.ceil(price / tick_size) * tick_size
                    # Add similar logic for TP if needed (round up Long TP, round down Short TP)

            return quantized_price

        except Exception as e:
            self.logger.error(f"Error quantizing price {price} for {self.symbol}: {e}", exc_info=True)
            return None

    def calculate_stop_loss(
        self, entry_price: Decimal, side: PositionSide, latest_indicators: pd.Series
    ) -> Optional[Decimal]:
        """Calculates the initial stop loss price based on ATR or fixed percentage."""
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
            self.logger.error(f"Invalid entry price for SL calculation: {entry_price}")
            return None

        sl_method = self.risk_config.get("stop_loss_method", "atr").lower()
        atr_multiplier = self.risk_config.get("atr_multiplier", Decimal("1.5"))
        fixed_percent = self.risk_config.get("fixed_stop_loss_percent", Decimal("2.0")) / Decimal("100")
        atr_period = self.indicator_config.get("atr_period", 14)
        atr_col = (
            f"ATRr_{atr_period}" if isinstance(atr_period, int) else None
        )  # pandas_ta default ATR name might be ATR_{period} or ATRr_{period} - check output

        # Double check ATR column name from pandas_ta output if needed
        if atr_col and atr_col not in latest_indicators.index:
            atr_col_alt = f"ATR_{atr_period}"
            if atr_col_alt in latest_indicators.index:
                atr_col = atr_col_alt
            else:
                self.logger.error(f"ATR column ({atr_col} or {atr_col_alt}) not found in indicators.")
                return None

        stop_loss_price = None

        if sl_method == "atr":
            if not atr_col:
                self.logger.error("ATR period not configured correctly.")
                return None
            if atr_col not in latest_indicators:
                self.logger.error(f"ATR column '{atr_col}' not found in latest indicators.")
                return None

            atr_value = latest_indicators[atr_col]
            if not isinstance(atr_value, Decimal) or not atr_value.is_finite() or atr_value <= 0:
                self.logger.error(f"Cannot calculate ATR stop loss: Invalid ATR value ({atr_value}).")
                return None

            stop_distance = atr_value * atr_multiplier
            if side == PositionSide.LONG:
                stop_loss_price = entry_price - stop_distance
            elif side == PositionSide.SHORT:
                stop_loss_price = entry_price + stop_distance

        elif sl_method == "fixed_percent":
            if fixed_percent <= 0 or fixed_percent >= 1:
                self.logger.error(
                    f"Invalid fixed_stop_loss_percent: {fixed_percent * 100}%. Must be between 0 and 100."
                )
                return None
            if side == PositionSide.LONG:
                stop_loss_price = entry_price * (Decimal("1") - fixed_percent)
            elif side == PositionSide.SHORT:
                stop_loss_price = entry_price * (Decimal("1") + fixed_percent)
        else:
            self.logger.error(f"Unknown stop loss method: {sl_method}")
            return None

        # Basic validation
        if stop_loss_price is None or not stop_loss_price.is_finite() or stop_loss_price <= 0:
            self.logger.error(
                f"Calculated stop loss price ({stop_loss_price}) is invalid (zero, negative, or non-finite). Cannot set SL."
            )
            return None

        # Quantize SL price to market precision (conservative rounding)
        quantized_sl = self.quantize_price(
            stop_loss_price, side=side
        )  # Pass side for conservative rounding if implemented

        if quantized_sl is None:
            self.logger.error("Failed to quantize stop loss price.")
            return None

        # Final check: Ensure quantized SL didn't cross entry price due to rounding
        if side == PositionSide.LONG and quantized_sl >= entry_price:
            self.logger.error(f"Quantized SL {quantized_sl} is >= entry price {entry_price}. Cannot set SL.")
            # Maybe adjust SL slightly further away? Requires careful thought. For now, fail.
            return None
        if side == PositionSide.SHORT and quantized_sl <= entry_price:
            self.logger.error(f"Quantized SL {quantized_sl} is <= entry price {entry_price}. Cannot set SL.")
            return None

        self.logger.info(f"Calculated Initial Stop Loss Price: {quantized_sl}")
        return quantized_sl

    def check_ma_cross_exit(self, latest_indicators: pd.Series, position_side: PositionSide) -> bool:
        """Checks if an MA cross exit condition is met."""
        if not self.trading_config.get("use_ma_cross_exit", False):
            return False

        ema_short_period = self.indicator_config.get("ema_short_period")
        ema_long_period = self.indicator_config.get("ema_long_period")
        ema_short_col = f"EMA_{ema_short_period}" if isinstance(ema_short_period, int) else None
        ema_long_col = f"EMA_{ema_long_period}" if isinstance(ema_long_period, int) else None

        if not ema_short_col or not ema_long_col:
            self.logger.warning("MA cross exit enabled, but EMA periods not configured correctly.")
            return False

        if not all(c in latest_indicators for c in [ema_short_col, ema_long_col]):
            self.logger.warning(
                f"Cannot check MA cross exit: EMA columns ({ema_short_col}, {ema_long_col}) not available in latest indicators."
            )
            return False

        ema_short = latest_indicators[ema_short_col]
        ema_long = latest_indicators[ema_long_col]

        if (
            not isinstance(ema_short, Decimal)
            or not ema_short.is_finite()
            or not isinstance(ema_long, Decimal)
            or not ema_long.is_finite()
        ):
            self.logger.warning(f"Invalid EMA values for MA cross exit check: Short={ema_short}, Long={ema_long}")
            return False

        # Check for bearish cross for long, bullish cross for short
        exit_signal = False
        if position_side == PositionSide.LONG and ema_short < ema_long:
            self.logger.info("MA Cross Exit triggered for LONG position (Short EMA crossed below Long EMA).")
            exit_signal = True
        elif position_side == PositionSide.SHORT and ema_short > ema_long:
            self.logger.info("MA Cross Exit triggered for SHORT position (Short EMA crossed above Long EMA).")
            exit_signal = True

        return exit_signal

    def manage_stop_loss(
        self, position: Dict[str, Any], latest_indicators: pd.Series, current_state: Dict[str, Any]
    ) -> Optional[Decimal]:
        """
        Manages stop loss adjustments (Break-Even, Trailing).
        Returns the new SL price if an update is needed, otherwise None.
        Requires 'active_position' and 'stop_loss_price' in current_state.
        """
        if not position or position.get("side") == "none" or position.get("side") is None:
            self.logger.debug("No active position or side is None, cannot manage SL.")
            return None  # No position to manage

        entry_price = position.get("entryPrice")  # Should be Decimal from fetch_positions
        current_sl_state = current_state.get("stop_loss_price")  # Get SL from our state (should be Decimal)
        position_side = PositionSide(position["side"])  # Convert string 'long'/'short' to Enum
        mark_price = position.get("markPrice")  # Should be Decimal

        # Validate necessary data
        if not all(isinstance(val, Decimal) and val.is_finite() for val in [entry_price, current_sl_state, mark_price]):
            self.logger.warning(
                f"Missing or invalid data for SL management: Entry={entry_price}, CurrentSL={current_sl_state}, MarkPrice={mark_price}"
            )
            return None
        if entry_price <= 0 or mark_price <= 0:
            self.logger.warning(
                f"Entry price or mark price is non-positive: Entry={entry_price}, Mark={mark_price}. Cannot manage SL."
            )
            return None

        new_sl_price = None
        state_updated_this_cycle = False  # Track if BE was set in this cycle

        # --- Get ATR value ---
        atr_period = self.indicator_config.get("atr_period", 14)
        atr_col = f"ATRr_{atr_period}" if isinstance(atr_period, int) else None
        # Double check ATR column name
        if atr_col and atr_col not in latest_indicators.index:
            atr_col_alt = f"ATR_{atr_period}"
            if atr_col_alt in latest_indicators.index:
                atr_col = atr_col_alt
            else:
                atr_col = None  # ATR not available

        atr_value = None
        if atr_col and atr_col in latest_indicators:
            val = latest_indicators[atr_col]
            if isinstance(val, Decimal) and val.is_finite() and val > 0:
                atr_value = val

        if atr_value is None:
            self.logger.warning("ATR value not available or invalid. Cannot perform ATR-based SL management (BE, TSL).")
            # If only ATR methods are enabled, we can't do anything.
            # If other methods exist, they might still run.

        # 1. Break-Even Stop Loss
        use_be = self.risk_config.get("use_break_even_sl", False)
        be_trigger_atr = self.risk_config.get("break_even_trigger_atr", Decimal("1.0"))
        be_offset_atr = self.risk_config.get("break_even_offset_atr", Decimal("0.1"))  # Small profit offset

        if use_be and atr_value is not None and not current_state.get("break_even_achieved", False):
            profit_target_distance = atr_value * be_trigger_atr
            offset_distance = atr_value * be_offset_atr

            if position_side == PositionSide.LONG:
                profit_target_price = entry_price + profit_target_distance
                be_price_with_offset = entry_price + offset_distance  # Target SL price slightly above entry
                if mark_price >= profit_target_price:
                    # Quantize potential BE price
                    quantized_be_price = self.quantize_price(be_price_with_offset, side=PositionSide.LONG)
                    if quantized_be_price is not None and quantized_be_price > current_sl_state:
                        self.logger.info(
                            f"Break-Even Triggered (Long): Mark Price {mark_price} >= Target {profit_target_price}. Moving SL from {current_sl_state} to {quantized_be_price}"
                        )
                        new_sl_price = quantized_be_price
                        current_state["break_even_achieved"] = True  # Mark BE achieved in state
                        state_updated_this_cycle = True
                    elif quantized_be_price is None:
                        self.logger.warning("Failed to quantize break-even price.")
                    elif quantized_be_price <= current_sl_state:
                        self.logger.debug(
                            f"BE triggered (Long), but proposed BE price {quantized_be_price} is not better than current SL {current_sl_state}."
                        )

            elif position_side == PositionSide.SHORT:
                profit_target_price = entry_price - profit_target_distance
                be_price_with_offset = entry_price - offset_distance  # Target SL price slightly below entry
                if mark_price <= profit_target_price:
                    # Quantize potential BE price
                    quantized_be_price = self.quantize_price(be_price_with_offset, side=PositionSide.SHORT)
                    if quantized_be_price is not None and quantized_be_price < current_sl_state:
                        self.logger.info(
                            f"Break-Even Triggered (Short): Mark Price {mark_price} <= Target {profit_target_price}. Moving SL from {current_sl_state} to {quantized_be_price}"
                        )
                        new_sl_price = quantized_be_price
                        current_state["break_even_achieved"] = True  # Mark BE achieved in state
                        state_updated_this_cycle = True
                    elif quantized_be_price is None:
                        self.logger.warning("Failed to quantize break-even price.")
                    elif quantized_be_price >= current_sl_state:
                        self.logger.debug(
                            f"BE triggered (Short), but proposed BE price {quantized_be_price} is not better than current SL {current_sl_state}."
                        )

        # 2. Trailing Stop Loss (only if BE wasn't just set in this cycle)
        use_tsl = self.risk_config.get("use_trailing_sl", False)
        tsl_atr_multiplier = self.risk_config.get("trailing_sl_atr_multiplier", Decimal("2.0"))

        if use_tsl and atr_value is not None and not state_updated_this_cycle:
            trail_distance = atr_value * tsl_atr_multiplier
            potential_tsl_price = None

            if position_side == PositionSide.LONG:
                potential_tsl_price = mark_price - trail_distance
                # Quantize potential TSL price (round down for long SL)
                quantized_tsl_price = self.quantize_price(potential_tsl_price, side=PositionSide.LONG)
                if quantized_tsl_price is not None and quantized_tsl_price > current_sl_state:
                    self.logger.debug(
                        f"Trailing SL Update (Long): Potential TSL {quantized_tsl_price} > Current SL {current_sl_state}"
                    )
                    new_sl_price = quantized_tsl_price  # Update SL if TSL is better
                elif quantized_tsl_price is None:
                    self.logger.warning("Failed to quantize trailing SL price (Long).")

            elif position_side == PositionSide.SHORT:
                potential_tsl_price = mark_price + trail_distance
                # Quantize potential TSL price (round up for short SL)
                quantized_tsl_price = self.quantize_price(potential_tsl_price, side=PositionSide.SHORT)
                if quantized_tsl_price is not None and quantized_tsl_price < current_sl_state:
                    self.logger.debug(
                        f"Trailing SL Update (Short): Potential TSL {quantized_tsl_price} < Current SL {current_sl_state}"
                    )
                    new_sl_price = quantized_tsl_price  # Update SL if TSL is better
                elif quantized_tsl_price is None:
                    self.logger.warning("Failed to quantize trailing SL price (Short).")

            if new_sl_price is not None and new_sl_price != current_sl_state:  # Check if TSL actually updated the price
                self.logger.info(f"Trailing Stop Loss Update: New SL Price calculated = {new_sl_price}")
                # Final check: Ensure TSL doesn't move SL unfavorably after BE achieved
                if current_state.get("break_even_achieved", False):
                    be_price_check = self.quantize_price(
                        entry_price, side=position_side
                    )  # Re-quantize entry for comparison
                    if be_price_check is not None:
                        if position_side == PositionSide.LONG and new_sl_price < be_price_check:
                            self.logger.warning(
                                f"TSL calculation resulted in SL ({new_sl_price}) below quantized entry ({be_price_check}) after BE. Clamping SL to entry."
                            )
                            new_sl_price = be_price_check
                        elif position_side == PositionSide.SHORT and new_sl_price > be_price_check:
                            self.logger.warning(
                                f"TSL calculation resulted in SL ({new_sl_price}) above quantized entry ({be_price_check}) after BE. Clamping SL to entry."
                            )
                            new_sl_price = be_price_check
                    else:
                        self.logger.warning("Could not quantize entry price for TSL vs BE check.")

        # --- Return the final new SL price if it's valid and different from current ---
        if new_sl_price is not None:
            # Final validation of the calculated new_sl_price
            if not new_sl_price.is_finite() or new_sl_price <= 0:
                self.logger.warning(f"Calculated new SL price ({new_sl_price}) is invalid. Ignoring update.")
                return None

            # Ensure SL is still valid relative to entry/mark price after all adjustments
            if position_side == PositionSide.LONG and new_sl_price >= mark_price:
                self.logger.warning(
                    f"New SL price {new_sl_price} is >= current mark price {mark_price}. Invalid SL. Ignoring update."
                )
                return None
            if position_side == PositionSide.SHORT and new_sl_price <= mark_price:
                self.logger.warning(
                    f"New SL price {new_sl_price} is <= current mark price {mark_price}. Invalid SL. Ignoring update."
                )
                return None

            # Only return if the new SL is meaningfully different from the current one
            # Use a small tolerance based on price precision if available
            price_tick_size = self.exchange.get_market(self.symbol).get("precision", {}).get("price")
            tolerance = Decimal(str(price_tick_size)) / 2 if price_tick_size else Decimal("1e-8")

            if abs(new_sl_price - current_sl_state) > tolerance:
                # Update state immediately before returning? No, let the caller update state upon successful API call.
                # current_state['stop_loss_price'] = new_sl_price # Don't update state here
                self.logger.info(f"Proposing SL update from {current_sl_state} to {new_sl_price}")
                return new_sl_price
            else:
                self.logger.debug(
                    f"Calculated new SL {new_sl_price} is not significantly different from current {current_sl_state}. No update needed."
                )
                return None  # No change needed

        return None  # No update needed


# --- Main Trading Bot Class ---


class TradingBot:
    """
    The main trading bot class orchestrating the fetch-analyze-execute loop.
    """

    def __init__(self, config_path: Path, state_path: Path):
        self.config_path = config_path
        self.state_path = state_path
        # Setup logger first, before loading config which might use logger
        self.logger = setup_logger()

        self.config = load_config(config_path, self.logger)
        if not self.config:
            self.logger.critical("Failed to load or validate configuration. Exiting.")
            sys.exit(1)  # Exit if config is invalid

        # Apply log level from config if present and valid
        global LOG_LEVEL
        log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
        log_level_enum = getattr(logging, log_level_str, None)
        if isinstance(log_level_enum, int):
            if LOG_LEVEL != log_level_enum:
                LOG_LEVEL = log_level_enum
                self.logger.setLevel(LOG_LEVEL)
                # Update handlers' levels too
                for handler in self.logger.handlers:
                    handler.setLevel(LOG_LEVEL)
                self.logger.info(f"Log level set to {log_level_str} from config.")
        else:
            self.logger.warning(
                f"Invalid log level '{log_level_str}' in config. Using default {logging.getLevelName(LOG_LEVEL)}."
            )

        self.state = load_state(state_path, self.logger)
        # Ensure essential state keys exist with default values
        self.state.setdefault("active_position", None)  # Stores dict of the current position managed by the bot
        self.state.setdefault("stop_loss_price", None)  # Stores Decimal SL price set by the bot
        self.state.setdefault("take_profit_price", None)  # Stores Decimal TP price set by the bot
        self.state.setdefault("break_even_achieved", False)  # Flag for BE SL activation
        self.state.setdefault("last_order_id", None)  # ID of the last order placed by the bot

        try:
            self.exchange = BybitV5Wrapper(self.config, self.logger)
            self.analyzer = TradingAnalyzer(self.config, self.logger)
            self.position_manager = PositionManager(self.config, self.logger, self.exchange)
        except Exception as e:
            self.logger.critical(f"Failed to initialize core components: {e}. Exiting.", exc_info=True)
            sys.exit(1)

        # Extract frequently used config values
        self.symbol = self.config["trading_settings"]["symbol"]
        self.timeframe = self.config["trading_settings"]["timeframe"]
        self.leverage = self.config["trading_settings"]["leverage"]  # Should be Decimal from loader
        self.quote_asset = self.config["trading_settings"]["quote_asset"]
        self.poll_interval = self.config["trading_settings"].get("poll_interval_seconds", 60)
        self.hedge_mode = self.config["trading_settings"].get("hedge_mode", False)

        self.is_running = True

    def run(self):
        """Starts the main trading loop."""
        self.logger.info("--- Starting Trading Bot ---")
        self.logger.info(
            f"Symbol: {self.symbol}, Timeframe: {self.timeframe}, Leverage: {self.leverage}x, Hedge Mode: {self.hedge_mode}"
        )
        if not self.initialize_exchange_settings():
            self.logger.critical("Failed to initialize exchange settings (e.g., leverage). Exiting.")
            sys.exit(1)

        # Initial sync before starting loop
        self.logger.info("Performing initial position sync...")
        initial_position = self.get_current_position()
        self.sync_bot_state_with_position(initial_position)
        save_state(self.state, self.state_path, self.logger)  # Save initial state

        while self.is_running:
            try:
                self.logger.info("--- New Trading Cycle ---")
                start_time = time.time()

                # 1. Fetch Data
                # Fetch slightly more data for indicator stability
                ohlcv_limit = self.config.get("indicator_settings", {}).get("ohlcv_fetch_limit", 250)
                ohlcv_data = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=ohlcv_limit)
                if ohlcv_data is None or ohlcv_data.empty:
                    self.logger.warning("Failed to fetch OHLCV data. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue

                # 2. Analyze Data
                indicators_df = self.analyzer.calculate_indicators(ohlcv_data)
                if indicators_df is None or indicators_df.empty:
                    self.logger.warning("Failed to calculate indicators. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue
                try:
                    latest_indicators = indicators_df.iloc[-1]
                except IndexError:
                    self.logger.warning("Indicators DataFrame is empty after calculation. Skipping cycle.")
                    self._wait_for_next_cycle(start_time)
                    continue

                signal, signal_details = self.analyzer.generate_signal(indicators_df)

                # 3. Fetch Current State (Balance, Position) - Fetch position again for latest data
                current_position = self.get_current_position()  # Fetches from exchange
                self.sync_bot_state_with_position(current_position)  # Update self.state['active_position'] etc.

                # 4. Decision Making & Execution
                if self.state.get("active_position"):
                    # Pass the fetched live position data to management functions
                    self.manage_existing_position(latest_indicators, signal, current_position)
                else:
                    # No active position according to state
                    self.attempt_new_entry(signal, latest_indicators, signal_details)

                # 5. Save State (potentially done within manage/entry/close methods upon success)
                # Save at end of cycle ensures latest state is persisted even if no action taken
                save_state(self.state, self.state_path, self.logger)

                # 6. Wait
                self._wait_for_next_cycle(start_time)

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt received. Stopping bot...")
                self.is_running = False
                # Perform any cleanup if needed (e.g., cancel open orders?)
            except ccxt.AuthenticationError:
                self.logger.critical("Authentication failed during main loop. Stopping bot.", exc_info=True)
                self.is_running = False
            except ccxt.NetworkError as e:
                self.logger.error(f"Network error in main loop: {e}. Retrying after delay.", exc_info=True)
                time.sleep(self.poll_interval)  # Wait standard interval after network error
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                # Consider more robust error handling (e.g., exponential backoff on certain errors)
                time.sleep(self.poll_interval * 2)  # Wait longer after an unexpected error

        self.logger.info("--- Trading Bot stopped ---")

    def initialize_exchange_settings(self) -> bool:
        """Set initial exchange settings like leverage and margin mode."""
        self.logger.info("Initializing exchange settings...")
        success = True

        # Set Leverage
        if not self.exchange.set_leverage(self.symbol, self.leverage):
            self.logger.error(
                f"Failed to set initial leverage to {self.leverage}x for {self.symbol}. Check permissions/settings."
            )
            success = False  # Mark as failure but continue if possible

        # Set Margin Mode (Example: Set to ISOLATED if not already)
        # margin_mode = self.config['trading_settings'].get('margin_mode', 'isolated').lower()
        # try:
        #     # CCXT method might be set_margin_mode or require private API call
        #     # Example using hypothetical set_margin_mode
        #     current_mode_info = self.exchange.fetch_position_mode(self.symbol) # Hypothetical
        #     if current_mode_info.get('marginMode') != margin_mode:
        #          self.logger.info(f"Setting margin mode to {margin_mode} for {self.symbol}...")
        #          # result = self.exchange.set_margin_mode(margin_mode, self.symbol, params={'category': self.exchange.category}) # Adjust call as needed
        #          # Check result
        #          self.logger.info(f"Margin mode set to {margin_mode}.")
        #     else:
        #          self.logger.info(f"Margin mode already set to {margin_mode}.")
        # except ccxt.NotSupported:
        #      self.logger.warning(f"Exchange does not support setting margin mode via CCXT for {self.symbol}.")
        # except Exception as e:
        #      self.logger.error(f"Failed to set margin mode to {margin_mode}: {e}", exc_info=True)
        #      success = False

        # Add other initializations if needed...

        return success

    def get_current_position(self) -> Optional[Dict[str, Any]]:
        """Fetches and returns the single active position for the bot's symbol, handling hedge mode."""
        positions = self.exchange.fetch_positions(self.symbol)
        if positions is None:
            self.logger.warning(f"Could not fetch positions for {self.symbol}.")
            return None  # Indicate failure to fetch

        active_positions = [
            p
            for p in positions
            if p.get("contracts", Decimal(0)) != Decimal(0) or p.get("info", {}).get("size", Decimal(0)) != Decimal(0)
        ]

        if not active_positions:
            self.logger.info(f"No active position found for {self.symbol}.")
            return None  # No position exists

        if self.hedge_mode:
            # In hedge mode, the bot logic currently manages one side based on its state.
            # Find the position matching the side/idx stored in the bot's state if a position exists.
            state_pos = self.state.get("active_position")
            if state_pos:
                target_idx = state_pos.get("position_idx")
                target_side = state_pos.get("side")  # PositionSide Enum
                found_match = None
                for p in active_positions:
                    p_idx = p.get("positionIdx")
                    p_side_str = p.get("side")  # 'long' or 'short' string
                    # Match based on positionIdx primarily
                    if p_idx is not None and p_idx == target_idx:
                        found_match = p
                        break
                    # Fallback: match based on side if idx missing/mismatched (less reliable)
                    elif p_side_str == target_side.value:
                        found_match = p
                        # Don't break here, keep looking for idx match if possible

                if found_match:
                    self.logger.info(
                        f"Found active hedge mode position matching state: Side {found_match.get('side')}, Idx {found_match.get('positionIdx')}"
                    )
                    return found_match
                else:
                    # Bot state has a position, but no matching one found on exchange. State is likely stale.
                    self.logger.warning(
                        f"Bot state indicates hedge position (Idx: {target_idx}, Side: {target_side}), but no matching active position found on exchange."
                    )
                    # Let sync_bot_state handle clearing the stale state later. Return None for now.
                    return None
            else:
                # Bot state has no position, but found active position(s) on exchange.
                # This indicates external position(s). Bot should ignore them until state is synced.
                self.logger.warning(
                    f"Found {len(active_positions)} active hedge mode position(s) on exchange, but bot state has no active position. Ignoring external positions."
                )
                return None  # Don't return external positions if bot thinks it has none

        else:  # Non-hedge mode
            if len(active_positions) > 1:
                self.logger.warning(
                    f"Expected only one active position for {self.symbol} in non-hedge mode, but found {len(active_positions)}. Using the first one found."
                )
                # Potentially refine this: choose largest, or one matching state if state exists?
            pos = active_positions[0]
            self.logger.info(
                f"Found active non-hedge position: Side {pos.get('side')}, Size {pos.get('contracts') or pos.get('info', {}).get('size')}"
            )
            return pos

    def sync_bot_state_with_position(self, current_position_on_exchange: Optional[Dict[str, Any]]):
        """
        Updates the bot's internal state ('active_position', 'stop_loss_price', etc.)
        based on the fetched position from the exchange. Clears state if position is gone.
        """
        bot_thinks_has_position = self.state.get("active_position") is not None

        if current_position_on_exchange:
            # Position exists on exchange
            exchange_pos_side_str = current_position_on_exchange.get("side")
            exchange_pos_size = current_position_on_exchange.get("contracts") or current_position_on_exchange.get(
                "info", {}
            ).get("size")
            exchange_pos_entry = current_position_on_exchange.get("entryPrice")
            exchange_pos_idx = current_position_on_exchange.get("positionIdx")  # Hedge mode index
            exchange_sl = current_position_on_exchange.get("info", {}).get("stopLoss")  # SL from exchange 'info'

            # Convert exchange SL to Decimal if present
            exchange_sl_dec = None
            if exchange_sl is not None and exchange_sl != "" and exchange_sl != "0":  # Check for valid SL string
                try:
                    exchange_sl_dec = Decimal(str(exchange_sl))
                except InvalidOperation:
                    self.logger.warning(f"Could not parse stop loss '{exchange_sl}' from exchange position info.")

            if not bot_thinks_has_position:
                # Bot thought no position, but found one. Sync up state.
                self.logger.warning(f"Found unexpected active position on exchange for {self.symbol}. Syncing state.")
                self.state["active_position"] = {
                    "symbol": current_position_on_exchange.get("symbol"),
                    "side": PositionSide(exchange_pos_side_str)
                    if exchange_pos_side_str in ["long", "short"]
                    else PositionSide.NONE,
                    "entry_price": exchange_pos_entry,
                    "size": exchange_pos_size,
                    "position_idx": exchange_pos_idx,  # Store for hedge mode
                }
                # Attempt to adopt the SL found on the exchange if it exists
                if exchange_sl_dec is not None and exchange_sl_dec > 0:
                    self.state["stop_loss_price"] = exchange_sl_dec
                    self.logger.info(f"Adopted stop loss {exchange_sl_dec} found on exchange position.")
                else:
                    self.state["stop_loss_price"] = None  # No SL found or invalid
                    self.logger.warning(
                        "No valid stop loss found on the unexpected exchange position. SL state is None."
                    )

                self.state["take_profit_price"] = None  # Assume no TP managed by bot initially
                self.state["break_even_achieved"] = False  # Reset BE state
                self.logger.info(
                    f"Bot state synced. Current position: {self.state['active_position']['side'].name} {self.state['active_position']['size']} @ {self.state['active_position']['entry_price']}. SL: {self.state['stop_loss_price']}"
                )

            else:
                # Bot already knew about a position. Verify and update details.
                state_pos = self.state["active_position"]
                state_side = state_pos["side"]
                state_idx = state_pos.get("position_idx")

                # Check if the exchange position matches the bot's state (side and idx for hedge)
                match = False
                if self.hedge_mode:
                    if state_side.value == exchange_pos_side_str and state_idx == exchange_pos_idx:
                        match = True
                else:  # Non-hedge: just check side
                    if state_side.value == exchange_pos_side_str:
                        match = True

                if match:
                    # Position matches state, update dynamic values like size, entry, SL
                    self.logger.debug("Exchange position matches state. Updating details.")
                    state_pos["size"] = exchange_pos_size
                    state_pos["entry_price"] = exchange_pos_entry  # Update entry if avg price changed
                    # Update SL state ONLY if exchange SL differs from state SL
                    # This prevents overwriting a TSL update that hasn't hit the exchange yet
                    current_state_sl = self.state.get("stop_loss_price")
                    if exchange_sl_dec is not None and exchange_sl_dec > 0:
                        if current_state_sl is None or abs(exchange_sl_dec - current_state_sl) > Decimal("1e-8"):
                            self.logger.info(
                                f"Detected SL change on exchange. Updating state SL from {current_state_sl} to {exchange_sl_dec}."
                            )
                            self.state["stop_loss_price"] = exchange_sl_dec
                            # If SL moved to entry or better after BE was expected, mark BE achieved
                            if not self.state.get("break_even_achieved", False):
                                entry = state_pos["entry_price"]
                                if state_side == PositionSide.LONG and exchange_sl_dec >= entry:
                                    self.state["break_even_achieved"] = True
                                    self.logger.info(
                                        "Marking Break-Even as achieved based on updated SL from exchange."
                                    )
                                elif state_side == PositionSide.SHORT and exchange_sl_dec <= entry:
                                    self.state["break_even_achieved"] = True
                                    self.logger.info(
                                        "Marking Break-Even as achieved based on updated SL from exchange."
                                    )
                    elif current_state_sl is not None:
                        # Exchange shows no SL, but state has one. Maybe it got cancelled?
                        self.logger.warning(
                            f"Bot state has SL {current_state_sl}, but no SL found on exchange position. Clearing state SL."
                        )
                        self.state["stop_loss_price"] = None
                        # Potentially try to re-apply SL here? Or wait for manage loop.

                else:
                    # Position exists, but doesn't match state (e.g., wrong side/idx). Treat as external.
                    self.logger.warning(
                        f"Found active position on exchange ({exchange_pos_side_str}, Idx: {exchange_pos_idx}) that does NOT match bot state ({state_side.value}, Idx: {state_idx}). Clearing bot state."
                    )
                    self._clear_position_state("Mismatch with exchange position")

        else:
            # No position on exchange
            if bot_thinks_has_position:
                # Bot thought there was a position, but it's gone. Clear state.
                self.logger.info(
                    f"Position for {self.symbol} no longer found on exchange (likely closed/stopped/liquidated). Clearing bot state."
                )
                self._clear_position_state("Position closed on exchange")

    def _clear_position_state(self, reason: str):
        """Internal helper to clear all position-related state variables."""
        self.logger.info(f"Clearing position state. Reason: {reason}")
        self.state["active_position"] = None
        self.state["stop_loss_price"] = None
        self.state["take_profit_price"] = None
        self.state["break_even_achieved"] = False
        # Keep last_order_id for potential reference, or clear it too?
        # self.state['last_order_id'] = None

    def manage_existing_position(
        self, latest_indicators: pd.Series, signal: Signal, live_position_data: Dict[str, Any]
    ):
        """
        Manages SL, TP, and potential exits for the active position.
        Uses live_position_data fetched just before this call.
        Updates self.state directly based on outcomes.
        """
        self.logger.info("Managing existing position...")
        if not self.state.get("active_position"):
            self.logger.warning("manage_existing_position called but state has no active position. Sync issue?")
            return
        if not live_position_data:
            self.logger.error("manage_existing_position called without live position data. Cannot proceed.")
            return

        position_state = self.state["active_position"]
        position_side = position_state["side"]  # PositionSide Enum

        # Ensure we have a valid SL price in state to manage from
        if self.state.get("stop_loss_price") is None:
            self.logger.warning("Cannot manage position: Stop loss price is missing from bot state.")
            # Should we try to set an initial SL based on current state? Risky.
            # Or fetch SL from exchange again? Done in sync. If still None, maybe close?
            # For now, skip management if SL state is missing.
            return

        # 1. Check for MA Cross Exit (if enabled)
        if self.position_manager.check_ma_cross_exit(latest_indicators, position_side):
            self.logger.info("MA Cross exit condition met. Closing position.")
            self.close_position("MA Cross Exit")
            return  # Exit management cycle after closing

        # 2. Manage Stop Loss (BE, TSL)
        # Pass live position data and current state to position manager
        new_sl_price = self.position_manager.manage_stop_loss(live_position_data, latest_indicators, self.state)

        if new_sl_price is not None:
            # Ensure the proposed SL is valid before attempting to set
            if (
                position_side == PositionSide.LONG and new_sl_price >= live_position_data.get("markPrice", new_sl_price)
            ) or (
                position_side == PositionSide.SHORT
                and new_sl_price <= live_position_data.get("markPrice", new_sl_price)
            ):
                self.logger.warning(
                    f"Proposed new SL {new_sl_price} is invalid relative to mark price {live_position_data.get('markPrice')}. Aborting SL update."
                )
            else:
                self.logger.info(f"Attempting to update stop loss on exchange to: {new_sl_price}")
                pos_idx = position_state.get("position_idx") if self.hedge_mode else None
                # Pass only the SL price to set_protection
                if self.exchange.set_protection(self.symbol, stop_loss=new_sl_price, position_idx=pos_idx):
                    self.logger.info(f"Stop loss updated successfully on exchange to {new_sl_price}.")
                    # Update state ONLY on successful exchange update
                    self.state["stop_loss_price"] = new_sl_price
                    # Save state immediately after successful SL update
                    save_state(self.state, self.state_path, self.logger)
                else:
                    self.logger.error(f"Failed to update stop loss on exchange to {new_sl_price}.")
                    # State SL remains unchanged if exchange update fails. Will retry next cycle.
                    # Do NOT update self.state['break_even_achieved'] if API call failed.

        # 3. Check for Signal-Based Exit (Opposing Signal)
        exit_on_signal = self.config["trading_settings"].get("exit_on_opposing_signal", True)
        should_exit = False
        if exit_on_signal:
            # Define opposing signals
            if position_side == PositionSide.LONG and signal in [Signal.SELL, Signal.STRONG_SELL]:
                self.logger.info(f"Opposing signal ({signal.name}) received for LONG position. Closing.")
                should_exit = True
            elif position_side == PositionSide.SHORT and signal in [Signal.BUY, Signal.STRONG_BUY]:
                self.logger.info(f"Opposing signal ({signal.name}) received for SHORT position. Closing.")
                should_exit = True

        if should_exit:
            self.close_position(f"Opposing Signal ({signal.name})")
            return  # Exit management cycle

        self.logger.debug("No exit conditions met in this cycle. Position remains open.")

    def attempt_new_entry(self, signal: Signal, latest_indicators: pd.Series, signal_details: Dict[str, Any]):
        """Attempts to enter a new position based on the signal and risk parameters."""
        if self.state.get("active_position"):
            self.logger.warning("Attempted entry when a position is already active in state. Skipping entry.")
            return

        self.logger.info("Checking for new entry opportunities...")

        entry_signal = False
        target_side = None
        order_side = None

        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            entry_signal = True
            target_side = PositionSide.LONG
            order_side = OrderSide.BUY
        elif signal in [Signal.SELL, Signal.STRONG_SELL]:
            entry_signal = True
            target_side = PositionSide.SHORT
            order_side = OrderSide.SELL

        if not entry_signal:
            self.logger.info(f"Signal is {signal.name}. No entry condition met.")
            return

        # --- Pre-computation before placing order ---
        # 1. Get Current Price (use last close as estimate, or fetch ticker)
        entry_price_estimate = latest_indicators.get("close")
        if (
            not isinstance(entry_price_estimate, Decimal)
            or not entry_price_estimate.is_finite()
            or entry_price_estimate <= 0
        ):
            self.logger.warning("Could not get valid close price from indicators. Fetching ticker...")
            try:
                ticker = self.exchange.safe_ccxt_call("fetch_ticker", self.symbol)
                if ticker and ticker.get("last"):
                    entry_price_estimate = Decimal(str(ticker["last"]))
                    if not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
                        raise ValueError("Invalid ticker price")
                    self.logger.info(f"Using ticker last price for entry estimate: {entry_price_estimate}")
                else:
                    self.logger.error("Failed to get valid last price from ticker. Cannot proceed with entry.")
                    return
            except Exception as e:
                self.logger.error(f"Failed to fetch or parse ticker price: {e}. Cannot proceed with entry.")
                return

        # 2. Calculate Stop Loss
        stop_loss_price = self.position_manager.calculate_stop_loss(
            entry_price_estimate, target_side, latest_indicators
        )
        if stop_loss_price is None:
            self.logger.error("Failed to calculate stop loss. Cannot determine position size or place entry order.")
            return

        # 3. Calculate Position Size
        balance_info = self.exchange.fetch_balance()
        if not balance_info or self.quote_asset not in balance_info:
            self.logger.error(f"Failed to fetch sufficient balance info for {self.quote_asset}. Cannot size position.")
            return
        # Use total equity for risk calculation
        available_equity = balance_info[self.quote_asset]
        if not isinstance(available_equity, Decimal) or not available_equity.is_finite() or available_equity <= 0:
            self.logger.error(
                f"Invalid available equity for {self.quote_asset}: {available_equity}. Cannot size position."
            )
            return

        position_size = self.position_manager.calculate_position_size(
            entry_price_estimate, stop_loss_price, available_equity, self.quote_asset
        )
        if position_size is None or position_size <= 0:
            self.logger.warning("Position size calculation failed or resulted in zero/negative size. No entry.")
            return

        # --- Place Entry Order ---
        self.logger.info(
            f"Attempting to place {order_side.value} market order for {position_size} {self.symbol} with calculated SL {stop_loss_price}"
        )

        # Prepare parameters for create_order
        # Bybit V5 allows setting SL/TP directly with the order
        order_params = {
            # Use string representation for prices in params
            "stopLoss": str(stop_loss_price),
            # 'takeProfit': str(take_profit_price), # Add if TP is calculated/used
            # Trigger prices by Mark Price (common default, confirm if needed)
            # 'slTriggerBy': 'MarkPrice',
            # 'tpTriggerBy': 'MarkPrice',
        }
        # Add positionIdx for hedge mode entry
        position_idx = None
        if self.hedge_mode:
            position_idx = 1 if order_side == OrderSide.BUY else 2
            order_params["positionIdx"] = position_idx

        order_result = self.exchange.create_order(
            symbol=self.symbol,
            order_type="market",  # Consider limit orders to control entry price?
            side=order_side,
            amount=position_size,
            price=None,  # Market order
            params=order_params,
        )

        if order_result and order_result.get("id"):
            order_id = order_result["id"]
            self.logger.info(
                f"Entry order placed successfully. Order ID: {order_id}. Waiting briefly for fill/confirmation..."
            )
            self.state["last_order_id"] = order_id

            # --- Post-Order Placement State Update ---
            # Assume market order fills quickly near the estimated price.
            # A more robust system would poll the order status until filled, get the actual fill price,
            # and then update the state. This is a simplification.
            self.state["active_position"] = {
                "symbol": self.symbol,
                "side": target_side,
                "entry_price": entry_price_estimate,  # Use estimate, actual fill price might differ
                "size": position_size,  # Use the requested size, actual filled size might differ slightly
                "order_id": order_id,
                "position_idx": position_idx,  # Store if hedge mode
            }
            self.state["stop_loss_price"] = stop_loss_price  # Store the SL we *intended* to set
            self.state["take_profit_price"] = None  # Reset TP state
            self.state["break_even_achieved"] = False  # Reset BE state
            save_state(self.state, self.state_path, self.logger)  # Save state immediately after potential entry
            self.logger.info(f"Bot state updated for new {target_side.name} position. Intended SL: {stop_loss_price}.")

            # Optional: Short delay and verification step
            time.sleep(self.config["trading_settings"].get("post_order_verify_delay_seconds", 3))  # Configurable delay
            self.logger.info("Verifying position and SL after entry order...")
            final_pos = self.get_current_position()
            if final_pos:
                self.sync_bot_state_with_position(final_pos)  # Sync state with actual data
                # Check if SL reported by exchange matches intended SL
                exchange_sl_str = final_pos.get("info", {}).get("stopLoss")
                exchange_sl_dec = None
                if exchange_sl_str and exchange_sl_str != "0":
                    try:
                        exchange_sl_dec = Decimal(exchange_sl_str)
                    except:
                        pass

                if exchange_sl_dec is not None and abs(exchange_sl_dec - stop_loss_price) < Decimal("1e-8"):
                    self.logger.info(f"Stop loss {stop_loss_price} confirmed on exchange position.")
                elif exchange_sl_dec is not None:
                    self.logger.warning(
                        f"Stop loss on exchange ({exchange_sl_dec}) differs from intended ({stop_loss_price}). State SL updated to exchange value."
                    )
                    self.state["stop_loss_price"] = exchange_sl_dec  # Trust exchange value after entry
                else:
                    self.logger.warning(
                        "Stop loss NOT found or is zero on exchange position after placing order with SL parameter. Attempting to set SL via set_protection."
                    )
                    # Retry setting SL using set_protection
                    pos_idx_retry = self.state["active_position"].get("position_idx") if self.hedge_mode else None
                    if not self.exchange.set_protection(
                        self.symbol, stop_loss=stop_loss_price, position_idx=pos_idx_retry
                    ):
                        self.logger.error("Retry failed to set stop loss after entry using set_protection.")
                    else:
                        self.logger.info("Successfully set stop loss via set_protection after initial attempt failed.")
                        self.state["stop_loss_price"] = (
                            stop_loss_price  # Update state SL to intended value after successful set
                        )
                save_state(self.state, self.state_path, self.logger)  # Save state again after verification/retry
            else:
                # Position not found after placing order? Order might have failed/rejected silently or filled and immediately closed?
                self.logger.error(
                    "Position not found on exchange shortly after placing entry order. Clearing potentially stale state."
                )
                self._clear_position_state("Position not found post-entry")
                save_state(self.state, self.state_path, self.logger)

        else:
            self.logger.error("Failed to place entry order (API call failed or returned invalid result).")
            # Ensure state remains clean if order failed
            self._clear_position_state("Entry order placement failed")
            save_state(self.state, self.state_path, self.logger)

    def close_position(self, reason: str):
        """Closes the current active position with a market order."""
        if not self.state.get("active_position"):
            self.logger.warning("Close position called but no active position in state.")
            return

        position_info = self.state["active_position"]
        size_to_close = position_info["size"]  # Should be Decimal
        current_side = position_info["side"]  # PositionSide Enum
        close_side = OrderSide.SELL if current_side == PositionSide.LONG else OrderSide.BUY

        if size_to_close <= 0:
            self.logger.error(
                f"Cannot close position: Size in state is zero or negative ({size_to_close}). Clearing state."
            )
            self._clear_position_state(f"Invalid size in state during close: {size_to_close}")
            save_state(self.state, self.state_path, self.logger)
            return

        self.logger.info(
            f"Attempting to close {current_side.name} position of size {size_to_close} for reason: {reason}"
        )

        # Parameters for closing order
        close_params = {
            "reduceOnly": True  # Crucial: ensure it only reduces/closes position
        }
        pos_idx = None
        if self.hedge_mode:
            # Need the correct positionIdx to close the specific side
            pos_idx = position_info.get("position_idx")
            if pos_idx is None:
                self.logger.error("Cannot close hedge mode position: positionIdx missing from state. Aborting close.")
                # Maybe try fetching position again to get idx? Risky.
                return
            close_params["positionIdx"] = pos_idx

        # Cancel existing TP/SL orders before closing? Bybit might do this automatically with reduceOnly market orders.
        # self.cancel_open_orders_for_symbol() # Optional: Implement cancellation if needed

        order_result = self.exchange.create_order(
            symbol=self.symbol,
            order_type="market",
            side=close_side,
            amount=size_to_close,  # Pass Decimal size
            price=None,
            params=close_params,
        )

        if order_result and order_result.get("id"):
            order_id = order_result["id"]
            self.logger.info(f"Position close order placed successfully. Order ID: {order_id}. Reason: {reason}")
            # Clear bot state immediately after placing close order
            # More robust: wait for fill confirmation before clearing state, but market orders usually fill fast.
            self._clear_position_state(f"Close order placed (ID: {order_id}, Reason: {reason})")
            self.state["last_order_id"] = order_id  # Store the close order ID
            save_state(self.state, self.state_path, self.logger)  # Save cleared state
        else:
            self.logger.error(
                f"Failed to place position close order for reason: {reason}. Position state remains unchanged."
            )
            # State remains unchanged, will likely retry closing on next cycle if conditions persist.

    def _wait_for_next_cycle(self, cycle_start_time: float):
        """Waits until the next polling interval, considering execution time."""
        cycle_end_time = time.time()
        execution_time = cycle_end_time - cycle_start_time
        wait_time = max(0, self.poll_interval - execution_time)
        self.logger.debug(f"Cycle execution time: {execution_time:.2f}s. Waiting for {wait_time:.2f}s...")
        if wait_time > 0:
            time.sleep(wait_time)


# --- Main Execution ---

if __name__ == "__main__":
    # Ensure Decimal context is set globally early
    getcontext().prec = DECIMAL_PRECISION + 4
    getcontext().rounding = ROUND_HALF_UP

    # Create bot instance (initializes logger, loads config/state, sets up exchange)
    try:
        bot = TradingBot(config_path=CONFIG_FILE, state_path=STATE_FILE)
        # Start the main loop
        bot.run()
    except SystemExit as e:
        # Logged critical errors already caused exit
        print(f"Bot exited with code {e.code}.")
    except Exception as e:
        # Catch any unexpected critical errors during setup or run
        # Logger might not be fully initialized here, so print as well
        print(f"CRITICAL UNHANDLED ERROR: {e}", file=sys.stderr)
        # Try logging if possible
        try:
            logging.getLogger("TradingBot").critical(f"Unhandled exception caused bot termination: {e}", exc_info=True)
        except Exception:
            pass  # Ignore logging errors if logger failed
        sys.exit(1)  # Exit with error code
