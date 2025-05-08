"""Scalping Bot v4 - Pyrmethus Enhanced Edition (Bybit V5 Optimized).

Implements an enhanced cryptocurrency scalping bot using ccxt, specifically
tailored for Bybit V5 API's parameter-based Stop Loss (SL) and Take Profit (TP)
handling. This version incorporates fixes for precision errors, improved state
management, enhanced logging, and more robust error handling based on provided logs
and code analysis.

Key Enhancements V4 (Compared to a hypothetical V3 / Original Provided Code):
- Addressed critical 'InvalidOrder' error related to amount precision/minimums
  in `calculate_order_size` by adding explicit checks against market limits AFTER
  applying precision.
- Enhanced `calculate_order_size` to catch precision/limit errors gracefully and
  provide clearer warnings (addressing the root cause of log errors).
- Made state file handling significantly more robust (`_load_state`, `_save_state`)
  with atomic writes (using temporary files and `os.replace`), automated backup
  on corruption, and better validation.
- Improved API retry logic (`@retry_api_call`) with more specific error handling
  (e.g., distinguishing more non-retryable ExchangeErrors, handling Auth errors).
- Enhanced logging: Added thread name, improved formatting, made file handler
  setup more robust, increased debug logging, used f-strings consistently.
- Refined `fetch_balance` to better handle Bybit V5 Unified/Contract/Spot accounts
  by checking multiple fields in the 'info' structure.
- Refined `_calculate_sl_tp_prices` with more robust sanity checks (e.g., vs. entry,
  SL vs. TP logic).
- Strengthened `_calculate_decimal_places` using logarithms and string fallbacks.
- Ensured consistent use of calculated `price_decimals` and `amount_decimals`
  for formatting logs and potentially prices/amounts sent to the API.
- Added clearer warnings and parameter fetching for the **EXPERIMENTAL** TSL feature
  using `edit_order` in `_manage_active_positions`.
- Improved PnL calculation logging in `_manage_active_positions`.
- Enhanced shutdown procedure (`shutdown`) to handle active/pending positions more clearly
  based on configuration.
- Added more detailed validation of configuration parameters in `validate_config`.
- General improvements to type hinting, comments, docstrings, and code clarity.
"""

import contextlib
import json
import logging
import math  # For potential decimal place calculation
import os
import shutil  # For state file backup and management
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import ccxt

# import ccxt.async_support as ccxt_async # Retained for potential future asynchronous implementation
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv

# --- Arcane Constants & Configuration Defaults ---
CONFIG_FILE_NAME: str = "config.yaml"
STATE_FILE_NAME: str = "scalping_bot_state.json"
LOG_FILE_NAME: str = "scalping_bot_v4.log"
DEFAULT_EXCHANGE_ID: str = "bybit"
DEFAULT_TIMEFRAME: str = "1m"
DEFAULT_RETRY_MAX: int = 3
DEFAULT_RETRY_DELAY_SECONDS: int = 3
DEFAULT_SLEEP_INTERVAL_SECONDS: int = 10
STRONG_SIGNAL_THRESHOLD_ABS: int = 3  # Absolute score threshold for strong signal adjustment
ENTRY_SIGNAL_THRESHOLD_ABS: int = 2  # Absolute score threshold to trigger entry
ATR_MULTIPLIER_SL: float = 2.0  # Default ATR multiplier for Stop Loss
ATR_MULTIPLIER_TP: float = 3.0  # Default ATR multiplier for Take Profit
DEFAULT_PRICE_DECIMALS: int = 4  # Fallback price decimal places
DEFAULT_AMOUNT_DECIMALS: int = 8  # Fallback amount decimal places


# Position Status Constants
STATUS_PENDING_ENTRY: str = "pending_entry"  # Order placed but not yet filled
STATUS_ACTIVE: str = "active"  # Order filled, position is live
STATUS_CLOSING: str = "closing"  # Manual close initiated (optional use)
STATUS_CANCELED: str = "canceled"  # Order explicitly cancelled
STATUS_CLOSED_ON_EXIT: str = "closed_on_exit"  # Position closed during bot shutdown
STATUS_UNKNOWN: str = "unknown"  # Order status cannot be determined

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# --- Centralized Logger Setup ---
logger = logging.getLogger("ScalpingBotV4")
logger.setLevel(logging.DEBUG)  # Set logger to lowest level to allow handlers to control verbosity
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s"  # Added brackets and thread name for clarity
)

# Console Handler (INFO level by default, configurable)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)  # Default console level
logger.addHandler(console_handler)

# File Handler (DEBUG level for detailed logs)
file_handler: logging.FileHandler | None = None
try:
    # Ensure log directory exists if LOG_FILE_NAME includes a path
    log_dir = os.path.dirname(LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Use print directly as logger might not be fully set up yet

    file_handler = logging.FileHandler(LOG_FILE_NAME, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log debug level to file
    logger.addHandler(file_handler)
except OSError:
    # Use print as logger file handler failed
    pass
except Exception:
    pass

# Load environment variables from .env file if present
if load_dotenv():
    logger.info(f"{Fore.CYAN}# Summoning secrets from .env scroll...{Style.RESET_ALL}")
else:
    logger.debug("No .env scroll found or no secrets contained within.")


# --- Robust API Retry Decorator ---
def retry_api_call(max_retries: int = DEFAULT_RETRY_MAX, initial_delay: int = DEFAULT_RETRY_DELAY_SECONDS) -> Callable:
    """Decorator for retrying CCXT API calls with exponential backoff.
    Handles common transient network errors, rate limits, and distinguishes
    non-retryable exchange errors like insufficient funds or invalid orders.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds between retries.

    Returns:
        A decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any | None:
            """Wrapper function that implements the retry logic."""
            retries = 0
            delay = initial_delay
            # The 'self' argument is args[0] if this decorates an instance method
            instance_name = args[0].__class__.__name__ if args and hasattr(args[0], "__class__") else ""
            func_name = f"{instance_name}.{func.__name__}" if instance_name else func.__name__

            while retries <= max_retries:
                try:
                    # Channeling the API call...
                    logger.debug(f"Attempting API call: {func_name} (Retry {retries}/{max_retries})")
                    return func(*args, **kwargs)

                # --- Specific Retryable CCXT Errors ---
                except ccxt.RateLimitExceeded as e:
                    logger.warning(
                        f"{Fore.YELLOW}Rate limit veil encountered ({func_name}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}) Error: {e}{Style.RESET_ALL}"
                    )
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                    # Log specific exception type for better debugging
                    logger.warning(
                        f"{Fore.YELLOW}Network ether disturbed ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )

                # --- Authentication / Permission Errors (Usually Fatal/Non-Retryable) ---
                except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
                    logger.critical(
                        f"{Fore.RED}Authentication/Permission ritual failed ({func_name}: {type(e).__name__} - {e}). "
                        f"Check API keys, permissions, IP whitelist. Aborting operation (no retry).{Style.RESET_ALL}"
                    )
                    return None  # Cannot recover without fixing credentials/permissions

                # --- Exchange Errors - Distinguish Retryable vs Non-Retryable ---
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # --- Non-Retryable / Fatal Conditions ---
                    # Order Lifecycle Related (Often expected/final states)
                    if any(
                        phrase in err_str
                        for phrase in [
                            "order not found",
                            "order does not exist",
                            "unknown order",
                            "order already cancelled",
                            "order was canceled",
                            "cancel order failed",  # Added variations
                            "order is finished",
                            "order has been filled",
                            "already closed",  # Variations of closed states
                            "cannot be modified",  # Order editing issues
                        ]
                    ):
                        # Log as WARNING because it's often expected (e.g., checking a closed order, cancelling filled order)
                        logger.warning(
                            f"{Fore.YELLOW}Order vanished, unknown, or final state encountered for {func_name}: {e}. Returning None (no retry).{Style.RESET_ALL}"
                        )
                        return None  # OrderNotFound or similar is often final for that specific call
                    # Funding / Margin Related
                    elif any(
                        phrase in err_str
                        for phrase in [
                            "insufficient balance",
                            "insufficient margin",
                            "margin is insufficient",
                            "available balance insufficient",
                            "insufficient funds",  # Added variations
                        ]
                    ):
                        logger.error(
                            f"{Fore.RED}Insufficient essence (funds/margin) for {func_name}: {e}. Aborting operation (no retry).{Style.RESET_ALL}"
                        )
                        return None  # Cannot recover without external action
                    # Invalid Request / Parameters / Limits / Config
                    elif any(
                        phrase in err_str
                        for phrase in [
                            "invalid order",
                            "parameter error",
                            "size too small",
                            "price too low",
                            "price too high",
                            "invalid price precision",
                            "invalid amount precision",
                            "api key is invalid",  # Handled by AuthenticationError, but keep just in case
                            "position mode not modified",
                            "risk limit",  # Config/Setup errors
                            "reduceonly",  # Often indicates trying to close more than open
                            "order cost not meet",  # Minimum cost issue
                            "bad symbol",
                            "invalid symbol",  # Config error
                            "leverage not modified",  # Config/setup error
                        ]
                    ):
                        logger.error(
                            f"{Fore.RED}Invalid parameters, limits, auth, or config for {func_name}: {e}. Aborting operation (no retry). Review logic/config/permissions.{Style.RESET_ALL}"
                        )
                        return None  # Requires code or config fix
                    # --- Potentially Retryable Exchange Errors ---
                    # These might include temporary glitches, server-side issues, nonce problems etc.
                    elif any(
                        phrase in err_str
                        for phrase in ["nonce", "timeout", "service unavailable", "internal error", "busy"]
                    ):
                        logger.warning(
                            f"{Fore.YELLOW}Potentially transient Exchange spirit troubled ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... "
                            f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                        )
                    # --- Unknown/Default Exchange Errors ---
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Unclassified Exchange spirit troubled ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... "
                            f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                        )

                # --- Catch-all for other unexpected exceptions ---
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Unexpected rift during {func_name}: {type(e).__name__} - {e}. Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}",
                        exc_info=True,  # Log traceback for unexpected errors
                    )

                # --- Retry Logic ---
                if retries < max_retries:
                    time.sleep(delay)
                    # Exponential backoff, capped at 60 seconds
                    delay = min(delay * 2, 60)
                    retries += 1
                else:
                    # Max retries reached, operation failed
                    logger.error(
                        f"{Fore.RED}Max retries ({max_retries}) reached for {func_name}. The spell falters.{Style.RESET_ALL}"
                    )
                    return None  # Indicate failure after exhausting retries

            # Should theoretically not be reached, but acts as a failsafe
            return None

        return wrapper

    return decorator


# --- Core Scalping Bot Class ---
class ScalpingBot:
    """Pyrmethus Enhanced Scalping Bot v4. Optimized for Bybit V5 API.

    This bot implements a scalping strategy using technical indicators and
    order book analysis. It features robust persistent state management, ATR-based or
    fixed percentage SL/TP, parameter-based SL/TP placement via Bybit V5 API,
    and an **EXPERIMENTAL** Trailing Stop Loss (TSL) mechanism using `edit_order`.

    **Disclaimer:** Trading cryptocurrencies involves significant risk. This bot
    is provided for educational purposes and experimentation. Use at your own risk.
    Thoroughly test in simulation and testnet modes before considering live trading.
    The TSL feature using `edit_order` requires careful verification and may not
    be supported correctly or reliably by your exchange or the current CCXT version.
    Consider alternatives like monitoring price and placing market/limit close orders
    for TSL if `edit_order` proves unreliable.
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME, state_file: str = STATE_FILE_NAME) -> None:
        """Initializes the bot instance.

        Loads configuration, validates settings, establishes exchange connection,
        loads persistent state, and caches necessary market information including
        price and amount decimal places.

        Args:
            config_file: Path to the YAML configuration file.
            state_file: Path to the JSON file for storing persistent state.
        """
        logger.info(f"{Fore.MAGENTA}--- Pyrmethus Scalping Bot v4 Awakening ---{Style.RESET_ALL}")
        self.config: dict[str, Any] = {}
        self.state_file: str = state_file
        self.load_config(config_file)
        self.validate_config()  # Validate before using config values

        # --- Bind Core Attributes from Config/Environment ---
        # Credentials (prefer environment variables for security)
        self.api_key: str | None = os.getenv("BYBIT_API_KEY")
        self.api_secret: str | None = os.getenv("BYBIT_API_SECRET")

        # Exchange Settings
        self.exchange_id: str = self.config["exchange"]["exchange_id"]
        self.testnet_mode: bool = self.config["exchange"]["testnet_mode"]  # Explicit flag for clarity

        # Trading Parameters
        self.symbol: str = self.config["trading"]["symbol"]
        # Internal simulation mode (runs logic without placing real orders)
        self.simulation_mode: bool = self.config["trading"]["simulation_mode"]
        self.entry_order_type: str = self.config["trading"]["entry_order_type"]
        self.limit_order_entry_offset_pct_buy: float = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_entry_offset_pct_sell: float = self.config["trading"]["limit_order_offset_sell"]
        self.timeframe: str = self.config["trading"]["timeframe"]
        self.close_positions_on_exit: bool = self.config["trading"]["close_positions_on_exit"]

        # Order Book Parameters
        self.order_book_depth: int = self.config["order_book"]["depth"]
        self.imbalance_threshold: float = self.config["order_book"]["imbalance_threshold"]

        # Indicator Parameters
        self.volatility_window: int = self.config["indicators"]["volatility_window"]
        self.volatility_multiplier: float = self.config["indicators"]["volatility_multiplier"]
        self.ema_period: int = self.config["indicators"]["ema_period"]
        self.rsi_period: int = self.config["indicators"]["rsi_period"]
        self.macd_short_period: int = self.config["indicators"]["macd_short_period"]
        self.macd_long_period: int = self.config["indicators"]["macd_long_period"]
        self.macd_signal_period: int = self.config["indicators"]["macd_signal_period"]
        self.stoch_rsi_period: int = self.config["indicators"]["stoch_rsi_period"]
        self.stoch_rsi_k_period: int = self.config["indicators"]["stoch_rsi_k_period"]
        self.stoch_rsi_d_period: int = self.config["indicators"]["stoch_rsi_d_period"]
        self.atr_period: int = self.config["indicators"]["atr_period"]

        # Risk Management Parameters
        self.use_atr_sl_tp: bool = self.config["risk_management"]["use_atr_sl_tp"]
        self.atr_sl_multiplier: float = self.config["risk_management"]["atr_sl_multiplier"]
        self.atr_tp_multiplier: float = self.config["risk_management"]["atr_tp_multiplier"]
        # Base percentages used if use_atr_sl_tp is False
        self.base_stop_loss_pct: float | None = self.config["risk_management"].get("stop_loss_percentage")
        self.base_take_profit_pct: float | None = self.config["risk_management"].get("take_profit_percentage")
        # Bybit V5 trigger price types (e.g., MarkPrice, LastPrice, IndexPrice)
        self.sl_trigger_by: str | None = self.config["risk_management"].get("sl_trigger_by")
        self.tp_trigger_by: str | None = self.config["risk_management"].get("tp_trigger_by")
        # Trailing Stop Loss settings
        self.enable_trailing_stop_loss: bool = self.config["risk_management"]["enable_trailing_stop_loss"]
        self.trailing_stop_loss_percentage: float | None = self.config["risk_management"].get(
            "trailing_stop_loss_percentage"
        )
        # Position and Exit settings
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.time_based_exit_minutes: int | None = self.config["risk_management"].get("time_based_exit_minutes")
        self.order_size_percentage: float = self.config["risk_management"]["order_size_percentage"]
        # Optional signal strength adjustment (kept simple)
        self.strong_signal_adjustment_factor: float = self.config["risk_management"]["strong_signal_adjustment_factor"]
        self.weak_signal_adjustment_factor: float = self.config["risk_management"]["weak_signal_adjustment_factor"]

        # --- Internal Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0  # Simple daily PnL tracker (reset manually or add logic)
        # Stores active and pending positions as a list of dictionaries.
        # Example structure:
        # {
        #     'id': str,                       # Exchange Order ID of the entry order
        #     'symbol': str,                   # Trading symbol
        #     'side': str,                     # 'buy' or 'sell'
        #     'size': float,                   # Current position size (base currency), updated on fill
        #     'original_size': float,          # Requested size at entry
        #     'entry_price': Optional[float],  # Avg fill price (updated on fill), None if pending
        #     'entry_time': Optional[float],   # Timestamp (seconds) of entry fill, None if pending
        #     'status': str,                   # e.g., STATUS_PENDING_ENTRY, STATUS_ACTIVE, STATUS_CANCELED
        #     'entry_order_type': str,         # 'market' or 'limit'
        #     'stop_loss_price': Optional[float], # Price level for SL (initially calculated, potentially updated)
        #     'take_profit_price': Optional[float],# Price level for TP (initially calculated)
        #     'confidence': int,               # Signal score at entry
        #     'trailing_stop_price': Optional[float], # Current activated TSL price
        #     'last_update_time': float        # Timestamp of the last state update
        # }
        self.open_positions: list[dict[str, Any]] = []
        self.market_info: dict[str, Any] | None = None  # Cache for market details (precision, limits)
        self.price_decimals: int = DEFAULT_PRICE_DECIMALS  # Calculated/retrieved price decimal places
        self.amount_decimals: int = DEFAULT_AMOUNT_DECIMALS  # Calculated/retrieved amount decimal places

        # --- Setup & Initialization Steps ---
        self._configure_logging_level()  # Apply logging level from config
        self.exchange: ccxt.Exchange = self._initialize_exchange()  # Connect to exchange
        self._load_market_info()  # Load and cache market details for the symbol, calculate decimals
        self._load_state()  # Load persistent state from file

        # --- Log Final Operating Modes ---
        sim_color = Fore.YELLOW if self.simulation_mode else Fore.CYAN
        test_color = Fore.YELLOW if self.testnet_mode else Fore.GREEN
        logger.warning(f"{sim_color}--- INTERNAL SIMULATION MODE: {self.simulation_mode} ---{Style.RESET_ALL}")
        logger.warning(f"{test_color}--- EXCHANGE TESTNET MODE: {self.testnet_mode} ---{Style.RESET_ALL}")

        if not self.simulation_mode:
            if not self.testnet_mode:
                logger.warning(
                    f"{Fore.RED}{Style.BRIGHT}--- WARNING: LIVE TRADING ON MAINNET ACTIVE ---{Style.RESET_ALL}"
                )
                logger.warning(
                    f"{Fore.RED}{Style.BRIGHT}--- Ensure configuration and risk parameters are correct! ---{Style.RESET_ALL}"
                )
            else:
                logger.warning(f"{Fore.YELLOW}--- LIVE TRADING ON TESTNET ACTIVE ---{Style.RESET_ALL}")
        else:
            logger.info(
                f"{Fore.CYAN}--- Running in full internal simulation mode. No real orders will be placed. ---{Style.RESET_ALL}"
            )

        logger.info(
            f"{Fore.CYAN}Scalping Bot V4 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}"
        )

    def _configure_logging_level(self) -> None:
        """Sets the console logging level based on the configuration file."""
        try:
            log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)

            # Set the specific bot logger level AND the console handler level
            # File handler remains at DEBUG regardless of config
            # logger.setLevel(log_level) # Let handlers control verbosity based on global level
            console_handler.setLevel(log_level)  # Console mirrors configured level

            logger.info(f"Console logging level enchanted to: {log_level_str}")
            # Log effective levels for verification
            logger.debug(f"Effective Bot Logger Level: {logging.getLevelName(logger.level)}")
            logger.debug(f"Effective Console Handler Level: {logging.getLevelName(console_handler.level)}")
            if file_handler:
                logger.debug(f"Effective File Handler Level: {logging.getLevelName(file_handler.level)}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error configuring logging level: {e}. Using default INFO.{Style.RESET_ALL}")
            # logger.setLevel(logging.INFO) # Reset if error
            console_handler.setLevel(logging.INFO)

    def _load_state(self) -> None:
        """Loads bot state (open positions) from the state file.
        Implements robust backup and recovery logic in case of file corruption.
        """
        logger.debug(f"Attempting to recall state from {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        try:
            if os.path.exists(self.state_file):
                # Check file size first to avoid loading large potentially corrupted files
                if os.path.getsize(self.state_file) == 0:
                    logger.warning(
                        f"{Fore.YELLOW}State file {self.state_file} is empty. Starting fresh.{Style.RESET_ALL}"
                    )
                    self.open_positions = []
                    self._save_state()  # Create a valid empty state file
                    return

                with open(self.state_file, encoding="utf-8") as f:
                    content = f.read()
                    # Check content again after reading
                    if not content.strip():
                        logger.warning(
                            f"{Fore.YELLOW}State file {self.state_file} contains only whitespace. Starting fresh.{Style.RESET_ALL}"
                        )
                        self.open_positions = []
                        self._save_state()  # Create valid empty file
                        return

                    saved_state = json.loads(content)
                    # Basic validation: Ensure it's a list (of position dicts)
                    if isinstance(saved_state, list):
                        # Optional: Add more rigorous validation of each position dict structure here
                        valid_positions = []
                        required_keys = {"id", "symbol", "side", "status"}  # Example minimal required keys
                        for i, pos in enumerate(saved_state):
                            if isinstance(pos, dict) and required_keys.issubset(pos.keys()):
                                # Convert numeric strings back if needed (though saving as number is better)
                                for key in [
                                    "size",
                                    "original_size",
                                    "entry_price",
                                    "entry_time",
                                    "stop_loss_price",
                                    "take_profit_price",
                                    "trailing_stop_price",
                                    "last_update_time",
                                ]:
                                    if key in pos and isinstance(pos[key], str):
                                        try:
                                            pos[key] = float(pos[key])
                                        except (ValueError, TypeError):
                                            pass  # Keep as string if conversion fails
                                valid_positions.append(pos)
                            else:
                                logger.warning(
                                    f"Ignoring invalid entry #{i + 1} in state file (not a valid position dict or missing keys): {str(pos)[:100]}..."
                                )  # Log truncated entry
                        self.open_positions = valid_positions
                        logger.info(
                            f"{Fore.GREEN}Recalled {len(self.open_positions)} valid position(s) from state file.{Style.RESET_ALL}"
                        )
                        # Attempt to remove backup if load was successful
                        if os.path.exists(state_backup_file):
                            try:
                                os.remove(state_backup_file)
                                logger.debug(f"Removed old state backup file: {state_backup_file}")
                            except OSError as remove_err:
                                logger.warning(
                                    f"Could not remove old state backup file {state_backup_file}: {remove_err}"
                                )
                    else:
                        raise ValueError(
                            f"Invalid state format in {self.state_file} - expected a list, got {type(saved_state)}."
                        )
            else:
                logger.info(f"No prior state file found ({self.state_file}). Beginning anew.")
                self.open_positions = []
                self._save_state()  # Create initial empty state file

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"{Fore.RED}Error decoding/validating state file {self.state_file}: {e}. Attempting recovery...{Style.RESET_ALL}"
            )
            # Attempt to restore from backup if it exists
            if os.path.exists(state_backup_file):
                logger.warning(
                    f"{Fore.YELLOW}Attempting to restore state from backup: {state_backup_file}{Style.RESET_ALL}"
                )
                try:
                    shutil.copy2(state_backup_file, self.state_file)  # copy2 preserves metadata
                    logger.info(
                        f"{Fore.GREEN}State successfully restored from backup. Retrying load...{Style.RESET_ALL}"
                    )
                    self._load_state()  # Retry loading the restored file (recursive call)
                    return  # Exit this attempt, the retry will handle it
                except Exception as restore_err:
                    logger.error(
                        f"{Fore.RED}Failed to restore state from backup {state_backup_file}: {restore_err}. Starting fresh.{Style.RESET_ALL}"
                    )
                    self.open_positions = []
                    self._save_state()  # Create a fresh state file
            else:
                # No backup exists, backup the corrupted file and start fresh
                corrupted_backup_path = f"{self.state_file}.corrupted_{int(time.time())}"
                logger.warning(
                    f"{Fore.YELLOW}No backup found. Backing up corrupted file to {corrupted_backup_path} and starting fresh.{Style.RESET_ALL}"
                )
                try:
                    if os.path.exists(self.state_file):  # Ensure file exists before copying
                        shutil.copy2(self.state_file, corrupted_backup_path)
                except Exception as backup_err:
                    logger.error(
                        f"{Fore.RED}Could not back up corrupted state file {self.state_file}: {backup_err}{Style.RESET_ALL}"
                    )
                # Start with empty state and save it
                self.open_positions = []
                self._save_state()

        except Exception as e:
            # Catch any other unexpected errors during state loading
            logger.critical(
                f"{Fore.RED}Fatal error loading state from {self.state_file}: {e}{Style.RESET_ALL}", exc_info=True
            )
            logger.warning(
                "Proceeding with an empty state due to critical load failure. Manual review of state file recommended."
            )
            self.open_positions = []

    def _save_state(self) -> None:
        """Saves the current bot state (list of open positions) to the state file.
        Uses a temporary file and atomic rename (`os.replace`) for robustness.
        Creates a backup (`.bak`) before replacing the main state file.
        """
        if not hasattr(self, "open_positions"):
            logger.error("Cannot save state: 'open_positions' attribute does not exist.")
            return

        logger.debug(f"Recording current state ({len(self.open_positions)} positions) to {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        # Use PID and timestamp for uniqueness in temp file name
        temp_state_file = f"{self.state_file}.tmp_{os.getpid()}_{int(time.time())}"

        try:
            # --- Serialize State Data ---
            # Convert potential complex types (like numpy types if used elsewhere) to standard types
            # Use default=str as a fallback, but explicit conversion is better if needed
            state_data_to_save = json.loads(json.dumps(self.open_positions, default=str))

            # --- Write to Temporary File ---
            with open(temp_state_file, "w", encoding="utf-8") as f:
                json.dump(state_data_to_save, f, indent=4)  # Use indent for readability

            # --- Create Backup of Current State (if exists) ---
            if os.path.exists(self.state_file):
                try:
                    # copy2 preserves more metadata than copyfile
                    shutil.copy2(self.state_file, state_backup_file)
                    logger.debug(f"Created state backup: {state_backup_file}")
                except Exception as backup_err:
                    logger.warning(
                        f"Could not create state backup {state_backup_file}: {backup_err}. Proceeding with overwrite cautiously."
                    )

            # --- Atomically Replace the Old State File ---
            # os.replace is atomic on POSIX and attempts atomic on Windows
            os.replace(temp_state_file, self.state_file)
            logger.debug(f"State recorded successfully to {self.state_file}.")

        except OSError as e:
            logger.error(f"{Fore.RED}Could not scribe state to {self.state_file} (IO/OS Error): {e}{Style.RESET_ALL}")
            # Clean up temporary file if it exists after an error
            if os.path.exists(temp_state_file):
                with contextlib.suppress(OSError):
                    os.remove(temp_state_file)
        except TypeError as e:
            logger.error(
                f"{Fore.RED}Could not serialize state data to JSON: {e}. Check for non-serializable types in open_positions.{Style.RESET_ALL}",
                exc_info=True,
            )
            # Clean up temporary file if it exists after an error
            if os.path.exists(temp_state_file):
                with contextlib.suppress(OSError):
                    os.remove(temp_state_file)
        except Exception as e:
            logger.error(
                f"{Fore.RED}An unexpected error occurred while recording state: {e}{Style.RESET_ALL}", exc_info=True
            )
            # Clean up temporary file if it exists after an error
            if os.path.exists(temp_state_file):
                with contextlib.suppress(OSError):
                    os.remove(temp_state_file)

    def load_config(self, config_file: str) -> None:
        """Loads configuration from the specified YAML file or creates a default one if not found."""
        try:
            with open(config_file, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            # Basic structure validation
            if not isinstance(self.config, dict):
                logger.critical(
                    f"{Fore.RED}Fatal: Config file {config_file} has invalid structure (must be a dictionary). Aborting.{Style.RESET_ALL}"
                )
                sys.exit(1)
            logger.info(f"{Fore.GREEN}Configuration spellbook loaded from {config_file}{Style.RESET_ALL}")

        except FileNotFoundError:
            logger.warning(f"{Fore.YELLOW}Configuration spellbook '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
                # Info message moved to create_default_config
            except Exception:
                # Error message moved to create_default_config
                pass  # Error already logged
            # Exit regardless of whether default creation succeeded, user must review/edit
            sys.exit(1)

        except yaml.YAMLError as e:
            logger.critical(
                f"{Fore.RED}Fatal: Error parsing spellbook {config_file}: {e}. Check YAML syntax. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}Fatal: Unexpected chaos loading configuration: {e}{Style.RESET_ALL}", exc_info=True
            )
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file with sensible defaults and explanations."""
        default_config = {
            "logging": {
                # Logging level for console output: DEBUG, INFO, WARNING, ERROR, CRITICAL
                "level": "INFO",
            },
            "exchange": {
                # Exchange ID from CCXT (e.g., 'bybit', 'binance', 'kraken')
                "exchange_id": os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE_ID),
                # Explicit flag for using testnet environment. Set to false for mainnet.
                "testnet_mode": os.getenv("TESTNET_MODE", "True").lower() in ("true", "1", "yes"),
                # API Credentials should ideally be set via environment variables:
                # BYBIT_API_KEY=YOUR_KEY
                # BYBIT_API_SECRET=YOUR_SECRET
                # If not using environment variables, uncomment and add here (less secure):
                # "api_key": "YOUR_API_KEY_HERE",
                # "api_secret": "YOUR_API_SECRET_HERE",
            },
            "trading": {
                # Symbol format depends on exchange and market type
                # Example Bybit USDT Perpetuals: BTC/USDT:USDT
                # Example Bybit USDC Perpetuals: BTC/USDC:USDC
                # Example Binance Futures: BTC/USDT
                "symbol": os.getenv("TRADING_SYMBOL", "DOT/USDT:USDT"),  # Changed default to DOT based on logs
                # Timeframe for OHLCV data (e.g., 1m, 5m, 1h) - Must be supported by exchange
                "timeframe": os.getenv("TIMEFRAME", DEFAULT_TIMEFRAME),
                # Internal simulation: if true, bot runs logic but places no real orders.
                "simulation_mode": os.getenv("SIMULATION_MODE", "True").lower() in ("true", "1", "yes"),
                # Entry order type: 'limit' or 'market'
                "entry_order_type": os.getenv("ENTRY_ORDER_TYPE", "limit").lower(),
                # Offset from current price for limit buy orders (e.g., 0.0005 = 0.05% below current)
                "limit_order_offset_buy": float(os.getenv("LIMIT_ORDER_OFFSET_BUY", 0.0005)),
                # Offset from current price for limit sell orders (e.g., 0.0005 = 0.05% above current)
                "limit_order_offset_sell": float(os.getenv("LIMIT_ORDER_OFFSET_SELL", 0.0005)),
                # Set to true to automatically close all open ACTIVE positions via market order on bot shutdown.
                # USE WITH CAUTION! Ensure this is the desired behavior. Pending orders are cancelled regardless.
                "close_positions_on_exit": False,
            },
            "order_book": {
                # Number of bid/ask levels to fetch
                "depth": 10,
                # Ask volume / Bid volume ratio threshold to indicate sell pressure
                # e.g., 1.5 means asks > 1.5 * bids = sell signal component
                # e.g., < (1 / 1.5) = 0.67 means asks < 0.67 * bids = buy signal component
                "imbalance_threshold": 1.5,
            },
            "indicators": {
                # --- Volatility ---
                "volatility_window": 20,  # Period for calculating rolling std dev of log returns
                "volatility_multiplier": 0.0,  # Multiplier for vol-based order size adjustment (0 = disabled, >0 = reduce size in high vol)
                # --- Moving Average ---
                "ema_period": 10,
                # --- Oscillators ---
                "rsi_period": 14,
                "stoch_rsi_period": 14,  # RSI period used within StochRSI
                "stoch_rsi_k_period": 3,  # Stochastic %K smoothing period (often 3)
                "stoch_rsi_d_period": 3,  # Stochastic %D smoothing period (often 3)
                # --- Trend/Momentum ---
                "macd_short_period": 12,
                "macd_long_period": 26,
                "macd_signal_period": 9,
                # --- Average True Range (for SL/TP and potentially size) ---
                "atr_period": 14,
            },
            "risk_management": {
                # Percentage of available quote balance to use for each new order (e.g., 0.05 = 5%)
                "order_size_percentage": 0.05,  # Increased default slightly
                # Maximum number of concurrent open positions allowed (active or pending entry)
                "max_open_positions": 1,
                # --- Stop Loss & Take Profit Method ---
                # Set to true to use ATR for SL/TP, false to use fixed percentages
                "use_atr_sl_tp": True,
                # --- ATR Settings (if use_atr_sl_tp is true) ---
                "atr_sl_multiplier": ATR_MULTIPLIER_SL,  # SL = entry_price +/- (ATR * multiplier)
                "atr_tp_multiplier": ATR_MULTIPLIER_TP,  # TP = entry_price +/- (ATR * multiplier)
                # --- Fixed Percentage Settings (if use_atr_sl_tp is false) ---
                # Percentage is applied to the entry price (e.g., 0.005 = 0.5%)
                "stop_loss_percentage": 0.005,
                "take_profit_percentage": 0.01,
                # --- Bybit V5 Trigger Price Types ---
                # Optional: Specify trigger type ('MarkPrice', 'LastPrice', 'IndexPrice'). Check Bybit/CCXT docs.
                # Defaults to exchange default (often MarkPrice) if not specified or None.
                "sl_trigger_by": "MarkPrice",  # Options: MarkPrice, LastPrice, IndexPrice, None
                "tp_trigger_by": "MarkPrice",  # Options: MarkPrice, LastPrice, IndexPrice, None
                # --- Trailing Stop Loss (EXPERIMENTAL - VERIFY SUPPORT & BEHAVIOR) ---
                # Uses edit_order which might not be reliable for SL/TP on all exchanges/CCXT versions.
                "enable_trailing_stop_loss": False,  # Enable/disable TSL feature (Default OFF due to experimental nature)
                # Trail distance as a percentage (e.g., 0.003 = 0.3%). Used only if TSL is enabled.
                "trailing_stop_loss_percentage": 0.003,
                # --- Other Exit Conditions ---
                # Time limit in minutes to close an active position if still open (0 or null/None to disable)
                "time_based_exit_minutes": 60,
                # --- Signal Adjustment (Optional - currently unused in signal score) ---
                # Multiplier applied to order size based on signal strength (1.0 = no adjustment)
                "strong_signal_adjustment_factor": 1.0,  # Applied if abs(score) >= STRONG_SIGNAL_THRESHOLD_ABS
                "weak_signal_adjustment_factor": 1.0,  # Applied if abs(score) < STRONG_SIGNAL_THRESHOLD_ABS
            },
        }
        # --- Environment Variable Overrides (Add more as needed) ---
        if os.getenv("ORDER_SIZE_PERCENTAGE"):
            try:
                override_val = float(os.environ["ORDER_SIZE_PERCENTAGE"])
                if 0 < override_val <= 1:
                    default_config["risk_management"]["order_size_percentage"] = override_val
                    logger.info(f"Overrode order_size_percentage from environment variable to {override_val}.")
                else:
                    logger.warning(
                        f"Invalid ORDER_SIZE_PERCENTAGE ({override_val}) in environment, must be > 0 and <= 1. Using default."
                    )
            except ValueError:
                logger.warning("Invalid ORDER_SIZE_PERCENTAGE in environment (not a number), using default.")

        try:
            # Ensure config directory exists if needed
            config_dir = os.path.dirname(config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                logger.info(f"Created config directory: {config_dir}")

            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(default_config, f, indent=4, sort_keys=False, default_flow_style=False)
            logger.info(f"{Fore.YELLOW}A default spellbook has been crafted: '{config_file}'.{Style.RESET_ALL}")
            logger.info(
                f"{Fore.YELLOW}IMPORTANT: Please review and tailor its enchantments (especially API keys in .env or config, symbol, risk settings), then restart the bot.{Style.RESET_ALL}"
            )
        except OSError as e:
            logger.error(f"{Fore.RED}Could not scribe default spellbook {config_file}: {e}{Style.RESET_ALL}")
            raise  # Re-raise to be caught by the calling function

    def validate_config(self) -> None:
        """Performs detailed validation of the loaded configuration parameters."""
        logger.debug("Scrutinizing the configuration spellbook for potency and validity...")
        try:
            # --- Helper Validation Functions ---
            def _get_nested(data: dict, keys: list[str], default: Any = None, req_type: type | None = None) -> Any:
                """Safely get nested dictionary values, optionally checking type."""
                value = data
                key_path = ""
                for key in keys:
                    key_path += f".{key}" if key_path else key
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        # logger.debug(f"Config check: Key path '{key_path}' not found, using default: {default}")
                        return default
                # Optional type check at the end
                if req_type is not None and not isinstance(value, req_type):
                    # Allow int to be treated as float if float is required
                    if req_type is float and isinstance(value, int):
                        return float(value)
                    raise TypeError(f"Expected type {req_type} but got {type(value)}")
                return value

            def _check(condition: bool, key_path: str, message: str) -> None:
                """Raise ValueError if condition is False."""
                if not condition:
                    raise ValueError(f"Config Error [{key_path}]: {message}")

            # --- Section Existence ---
            required_top_level_keys = {"logging", "exchange", "trading", "order_book", "indicators", "risk_management"}
            missing_sections = required_top_level_keys - set(self.config.keys())
            _check(not missing_sections, ".", f"Missing required configuration sections: {missing_sections}")
            for section in required_top_level_keys:
                _check(isinstance(self.config[section], dict), section, "Must be a dictionary (mapping).")

            # --- Logging Section ---
            cfg_log = self.config["logging"]
            log_level = _get_nested(cfg_log, ["level"], default="INFO", req_type=str).upper()
            _check(
                hasattr(logging, log_level),
                "logging.level",
                f"Invalid level '{log_level}'. Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
            )

            # --- Exchange Section ---
            cfg_ex = self.config["exchange"]
            ex_id = _get_nested(cfg_ex, ["exchange_id"], req_type=str)
            _check(ex_id and isinstance(ex_id, str), "exchange.exchange_id", "Must be a non-empty string.")
            _check(
                ex_id in ccxt.exchanges,
                "exchange.exchange_id",
                f"Unsupported exchange_id: '{ex_id}'. Available examples: {ccxt.exchanges[:10]}...",
            )
            _check(
                isinstance(_get_nested(cfg_ex, ["testnet_mode"]), bool),
                "exchange.testnet_mode",
                "Must be true or false.",
            )
            # API keys checked later during exchange initialization if needed

            # --- Trading Section ---
            cfg_tr = self.config["trading"]
            symbol = _get_nested(cfg_tr, ["symbol"], req_type=str)
            _check(symbol and isinstance(symbol, str), "trading.symbol", "Must be a non-empty string.")
            timeframe = _get_nested(cfg_tr, ["timeframe"], req_type=str)
            _check(timeframe and isinstance(timeframe, str), "trading.timeframe", "Must be a non-empty string.")
            # Note: Timeframe/Symbol validity checked later against exchange capabilities
            _check(
                isinstance(_get_nested(cfg_tr, ["simulation_mode"]), bool),
                "trading.simulation_mode",
                "Must be true or false.",
            )
            entry_type = _get_nested(cfg_tr, ["entry_order_type"], req_type=str).lower()
            _check(
                entry_type in ["market", "limit"],
                "trading.entry_order_type",
                f"Must be 'market' or 'limit', got: {entry_type}",
            )
            offset_buy = _get_nested(cfg_tr, ["limit_order_offset_buy"], req_type=(int, float))
            _check(
                isinstance(offset_buy, (int, float)) and offset_buy >= 0,
                "trading.limit_order_offset_buy",
                "Must be a non-negative number.",
            )
            offset_sell = _get_nested(cfg_tr, ["limit_order_offset_sell"], req_type=(int, float))
            _check(
                isinstance(offset_sell, (int, float)) and offset_sell >= 0,
                "trading.limit_order_offset_sell",
                "Must be a non-negative number.",
            )
            _check(
                isinstance(_get_nested(cfg_tr, ["close_positions_on_exit"]), bool),
                "trading.close_positions_on_exit",
                "Must be true or false.",
            )

            # --- Order Book Section ---
            cfg_ob = self.config["order_book"]
            depth = _get_nested(cfg_ob, ["depth"], req_type=int)
            _check(isinstance(depth, int) and depth > 0, "order_book.depth", "Must be a positive integer.")
            imbalance = _get_nested(cfg_ob, ["imbalance_threshold"], req_type=(int, float))
            _check(
                isinstance(imbalance, (int, float)) and imbalance > 0,
                "order_book.imbalance_threshold",
                "Must be a positive number.",
            )

            # --- Indicators Section ---
            cfg_ind = self.config["indicators"]
            period_keys = [
                "volatility_window",
                "ema_period",
                "rsi_period",
                "macd_short_period",
                "macd_long_period",
                "macd_signal_period",
                "stoch_rsi_period",
                "stoch_rsi_k_period",
                "stoch_rsi_d_period",
                "atr_period",
            ]
            for key in period_keys:
                val = _get_nested(cfg_ind, [key], req_type=int)
                _check(isinstance(val, int) and val > 0, f"indicators.{key}", "Must be a positive integer.")
            vol_mult = _get_nested(cfg_ind, ["volatility_multiplier"], req_type=(int, float))
            _check(
                isinstance(vol_mult, (int, float)) and vol_mult >= 0,
                "indicators.volatility_multiplier",
                "Must be a non-negative number.",
            )
            _check(
                cfg_ind["macd_short_period"] < cfg_ind["macd_long_period"],
                "indicators",
                "macd_short_period must be less than macd_long_period.",
            )

            # --- Risk Management Section ---
            cfg_rm = self.config["risk_management"]
            use_atr = _get_nested(cfg_rm, ["use_atr_sl_tp"], req_type=bool)
            _check(isinstance(use_atr, bool), "risk_management.use_atr_sl_tp", "Must be true or false.")
            if use_atr:
                atr_sl_mult = _get_nested(cfg_rm, ["atr_sl_multiplier"], req_type=(int, float))
                _check(
                    isinstance(atr_sl_mult, (int, float)) and atr_sl_mult > 0,
                    "risk_management.atr_sl_multiplier",
                    "Must be a positive number when use_atr_sl_tp is true.",
                )
                atr_tp_mult = _get_nested(cfg_rm, ["atr_tp_multiplier"], req_type=(int, float))
                _check(
                    isinstance(atr_tp_mult, (int, float)) and atr_tp_mult > 0,
                    "risk_management.atr_tp_multiplier",
                    "Must be a positive number when use_atr_sl_tp is true.",
                )
            else:
                sl_pct = _get_nested(cfg_rm, ["stop_loss_percentage"], req_type=(int, float))
                tp_pct = _get_nested(cfg_rm, ["take_profit_percentage"], req_type=(int, float))
                _check(
                    sl_pct is not None and isinstance(sl_pct, (int, float)) and 0 < sl_pct < 1,
                    "risk_management.stop_loss_percentage",
                    "Must be a number between 0 and 1 (exclusive, e.g., 0.01 for 1%) when use_atr_sl_tp is false.",
                )
                _check(
                    tp_pct is not None and isinstance(tp_pct, (int, float)) and 0 < tp_pct < 1,
                    "risk_management.take_profit_percentage",
                    "Must be a number between 0 and 1 (exclusive) when use_atr_sl_tp is false.",
                )

            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None]
            sl_trig = _get_nested(cfg_rm, ["sl_trigger_by"])
            tp_trig = _get_nested(cfg_rm, ["tp_trigger_by"])
            _check(
                sl_trig in valid_triggers,
                "risk_management.sl_trigger_by",
                f"Invalid trigger type: '{sl_trig}'. Must be one of: {valid_triggers}",
            )
            _check(
                tp_trig in valid_triggers,
                "risk_management.tp_trigger_by",
                f"Invalid trigger type: '{tp_trig}'. Must be one of: {valid_triggers}",
            )

            enable_tsl = _get_nested(cfg_rm, ["enable_trailing_stop_loss"], req_type=bool)
            _check(isinstance(enable_tsl, bool), "risk_management.enable_trailing_stop_loss", "Must be true or false.")
            if enable_tsl:
                tsl_pct = _get_nested(cfg_rm, ["trailing_stop_loss_percentage"], req_type=(int, float))
                _check(
                    tsl_pct is not None and isinstance(tsl_pct, (int, float)) and 0 < tsl_pct < 1,
                    "risk_management.trailing_stop_loss_percentage",
                    "Must be a positive number less than 1 (e.g., 0.005 for 0.5%) when TSL is enabled.",
                )

            order_size_pct = _get_nested(cfg_rm, ["order_size_percentage"], req_type=(int, float))
            _check(
                isinstance(order_size_pct, (int, float)) and 0 < order_size_pct <= 1,
                "risk_management.order_size_percentage",
                "Must be a number between 0 (exclusive) and 1 (inclusive, e.g., 0.05 for 5%).",
            )
            max_pos = _get_nested(cfg_rm, ["max_open_positions"], req_type=int)
            _check(
                isinstance(max_pos, int) and max_pos > 0,
                "risk_management.max_open_positions",
                "Must be a positive integer.",
            )
            time_exit = _get_nested(cfg_rm, ["time_based_exit_minutes"])
            _check(
                time_exit is None or (isinstance(time_exit, int) and time_exit >= 0),
                "risk_management.time_based_exit_minutes",
                "Must be a non-negative integer or null/None.",
            )
            strong_adj = _get_nested(cfg_rm, ["strong_signal_adjustment_factor"], req_type=(int, float))
            _check(
                isinstance(strong_adj, (int, float)) and strong_adj > 0,
                "risk_management.strong_signal_adjustment_factor",
                "Must be a positive number.",
            )
            weak_adj = _get_nested(cfg_rm, ["weak_signal_adjustment_factor"], req_type=(int, float))
            _check(
                isinstance(weak_adj, (int, float)) and weak_adj > 0,
                "risk_management.weak_signal_adjustment_factor",
                "Must be a positive number.",
            )

            logger.info(f"{Fore.GREEN}Configuration spellbook deemed valid and potent.{Style.RESET_ALL}")

        except (ValueError, TypeError) as e:
            logger.critical(
                f"{Fore.RED}Configuration flaw detected: {e}. Mend the '{CONFIG_FILE_NAME}' scroll. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}Unexpected chaos during configuration validation: {e}{Style.RESET_ALL}", exc_info=True
            )
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and configures the CCXT exchange instance."""
        logger.info(f"Opening communication channel with {self.exchange_id.upper()}...")

        # Check credentials early, especially if not in pure simulation mode
        # Prioritize environment variables, fallback to config (less secure)
        self.api_key = os.getenv("BYBIT_API_KEY") or self.config.get("exchange", {}).get("api_key")
        self.api_secret = os.getenv("BYBIT_API_SECRET") or self.config.get("exchange", {}).get("api_secret")
        creds_found = self.api_key and self.api_secret

        if not self.simulation_mode and not creds_found:
            logger.critical(
                f"{Fore.RED}API Key/Secret essence missing (check environment variables BYBIT_API_KEY/SECRET or config file). Cannot trade live or on testnet. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        elif creds_found and self.simulation_mode:
            logger.warning(
                f"{Fore.YELLOW}API Key/Secret found, but internal simulation_mode is True. Credentials will NOT be used for placing orders.{Style.RESET_ALL}"
            )
        elif not creds_found and self.simulation_mode:
            logger.info(
                f"{Fore.CYAN}Running in internal simulation mode. API credentials not required or found.{Style.RESET_ALL}"
            )

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            # Base configuration for the exchange instance
            exchange_config = {
                "enableRateLimit": True,  # Essential for respecting API limits
                "options": {
                    # Default market type (adjust based on typical use for the exchange)
                    "defaultType": "swap" if "bybit" in self.exchange_id.lower() else "future",
                    "adjustForTimeDifference": True,  # Corrects minor clock drifts
                    # Specific options can be added, e.g., for Bybit V5:
                    # 'recvWindow': 10000, # Example: Increase receive window if timeouts occur
                },
            }
            # Add Bybit V5 specific option if needed (check if CCXT handles this automatically)
            # if self.exchange_id.lower() == 'bybit':
            #     exchange_config['options']['defaultSubType'] = 'linear' # Or 'inverse'

            # Add API credentials only if needed (i.e., not internal simulation)
            if not self.simulation_mode:
                exchange_config["apiKey"] = self.api_key
                exchange_config["secret"] = self.api_secret
                logger.debug("API credentials added to exchange configuration.")

            exchange = exchange_class(exchange_config)

            # Set testnet mode using CCXT's unified method (if not in internal simulation)
            if not self.simulation_mode and self.testnet_mode:
                try:
                    logger.info("Attempting to switch exchange instance to Testnet mode...")
                    exchange.set_sandbox_mode(True)
                    logger.info(f"{Fore.YELLOW}Exchange sandbox mode explicitly enabled via CCXT.{Style.RESET_ALL}")
                except ccxt.NotSupported:
                    logger.warning(
                        f"{Fore.YELLOW}Exchange {self.exchange_id} does not support unified set_sandbox_mode via CCXT. Testnet functionality depends on API key type and exchange base URL (check Bybit key settings).{Style.RESET_ALL}"
                    )
                except Exception as sandbox_err:
                    logger.error(
                        f"{Fore.RED}Error setting sandbox mode: {sandbox_err}. Continuing, but testnet may not be active.{Style.RESET_ALL}"
                    )
            elif not self.simulation_mode and not self.testnet_mode:
                logger.info(f"{Fore.GREEN}Exchange sandbox mode is OFF. Operating on Mainnet.{Style.RESET_ALL}")

            # Load markets to get symbol info, limits, precision, etc.
            logger.debug("Loading market matrix...")
            exchange.load_markets(reload=True)  # Force reload to get latest info
            logger.debug("Market matrix loaded.")

            # Validate timeframe against loaded exchange capabilities
            if self.timeframe not in exchange.timeframes:
                available_tf = list(exchange.timeframes.keys())
                logger.critical(
                    f"{Fore.RED}Timeframe '{self.timeframe}' not supported by {self.exchange_id}. Available options start with: {available_tf[:15]}... Aborting.{Style.RESET_ALL}"
                )
                sys.exit(1)

            # Validate symbol against loaded markets
            if self.symbol not in exchange.markets:
                available_sym = list(exchange.markets.keys())
                logger.critical(
                    f"{Fore.RED}Symbol '{self.symbol}' not found on {self.exchange_id}. Available symbols start with: {available_sym[:10]}... Check config. Aborting.{Style.RESET_ALL}"
                )
                sys.exit(1)

            # Log confirmation of symbol and timeframe validity
            logger.info(
                f"Symbol '{self.symbol}' and timeframe '{self.timeframe}' confirmed available on {self.exchange_id}."
            )

            # Optional: Check if the specific market is active/tradeable
            market_details = exchange.market(self.symbol)
            if not market_details.get("active", True):  # Default to True if 'active' key is missing
                logger.warning(
                    f"{Fore.YELLOW}Warning: Market '{self.symbol}' is marked as inactive by the exchange.{Style.RESET_ALL}"
                )

            # Perform API connectivity/authentication test (if not simulating)
            if not self.simulation_mode:
                logger.debug("Performing initial API connectivity test (fetching time)...")

                # Use retry decorator for this initial check as well

                @retry_api_call(max_retries=1, initial_delay=1)
                def _test_connectivity(exchange_instance):
                    return exchange_instance.fetch_time()

                server_time_ms = _test_connectivity(exchange)

                if server_time_ms is not None:
                    server_time_str = pd.to_datetime(server_time_ms, unit="ms", utc=True).isoformat()
                    logger.debug(f"Exchange time crystal synchronized: {server_time_str}")
                    logger.info(f"{Fore.GREEN}API connection and authentication successful.{Style.RESET_ALL}")
                else:
                    # If fetch_time fails after retries, it's likely an auth or connection issue
                    logger.critical(
                        f"{Fore.RED}Initial API connectivity test (fetch_time) failed. Check credentials, permissions, network, and exchange status. Aborting.{Style.RESET_ALL}"
                    )
                    sys.exit(1)

            logger.info(
                f"{Fore.GREEN}Connection established and configured for {self.exchange_id.upper()}.{Style.RESET_ALL}"
            )
            return exchange

        except ccxt.AuthenticationError as e:
            # This might be caught here if it happens during initial class instantiation
            logger.critical(
                f"{Fore.RED}Authentication failed for {self.exchange_id.upper()}: {e}. Check API keys/permissions in .env file or config. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
            logger.critical(
                f"{Fore.RED}Could not connect to {self.exchange_id.upper()} ({type(e).__name__}): {e}. Check network/exchange status. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except ccxt.NotSupported as e:
            logger.critical(
                f"{Fore.RED}Exchange {self.exchange_id} reported 'Not Supported' during initialization: {e}. Configuration might be incompatible. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except Exception as e:
            # Catch-all for unexpected initialization errors
            logger.critical(
                f"{Fore.RED}Unexpected error initializing exchange {self.exchange_id.upper()}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _calculate_decimal_places(self, tick_size: float | int | None) -> int:
        """Calculates the number of decimal places from a tick size (e.g., 0.01 -> 2).
        Handles various formats including powers of 10 and other steps.

        Args:
            tick_size: The minimum price or amount increment. Can be None.

        Returns:
            The number of decimal places, or a default if calculation fails or tick_size is None.
        """
        fallback_decimals = DEFAULT_PRICE_DECIMALS  # Use a reasonable fallback

        if tick_size is None:
            logger.debug("Tick size is None, cannot calculate decimal places. Returning default.")
            return fallback_decimals
        if not isinstance(tick_size, (float, int)) or tick_size <= 0:
            logger.warning(f"Invalid tick size type or value for decimal calculation: {tick_size}. Returning default.")
            return fallback_decimals

        try:
            # Handle integer tick sizes (e.g., 1, 10) -> 0 decimals
            if isinstance(tick_size, int) or (isinstance(tick_size, float) and tick_size.is_integer()):
                return 0

            # Use logarithm for simple powers of 10 (e.g., 0.1, 0.001)
            log_val = math.log10(tick_size)
            if log_val == math.floor(log_val):  # Check if it's an integer (negative for < 1)
                decimals = max(0, int(-log_val))
                logger.debug(f"Calculated decimals for power-of-10 tick size {tick_size}: {decimals}")
                return decimals
            else:
                # If not a power of 10 (e.g., 0.5, 0.0025), convert to string and count
                # Use sufficient precision in formatting, remove trailing zeros
                s = format(tick_size, ".16f").rstrip("0")
                if "." in s:
                    num_decimals = len(s.split(".")[-1])
                    logger.debug(f"Calculated decimals via string format for tick size {tick_size}: {num_decimals}")
                    return num_decimals
                else:  # Should not happen if tick_size is float and not integer, but handle just in case
                    logger.warning(
                        f"Unexpected float tick size format: {tick_size}. Could not find decimal point in formatted string '{s}'. Returning default."
                    )
                    return fallback_decimals
        except ValueError as e:  # Handles log10(negative) which is already checked, but good practice
            logger.error(f"Math error calculating decimal places for tick size {tick_size}: {e}. Returning default.")
            return fallback_decimals
        except Exception as e:
            logger.error(
                f"Unexpected error calculating decimal places for tick size {tick_size}: {e}. Returning default."
            )
            return fallback_decimals

    def _load_market_info(self) -> None:
        """Loads and caches essential market information (precision, limits) for the trading symbol.
        Crucially, calculates and stores the number of decimal places for price and amount.
        """
        logger.debug(f"Loading market details for {self.symbol}...")
        try:
            # Use exchange.market() which gets info from already loaded markets
            self.market_info = self.exchange.market(self.symbol)
            if not self.market_info:
                raise ValueError(
                    f"Market info for symbol '{self.symbol}' could not be retrieved after loading markets."
                )

            # --- Extract Precision and Limits ---
            precision = self.market_info.get("precision", {})
            limits = self.market_info.get("limits", {})
            amount_tick = precision.get("amount")  # Minimum amount step (e.g., 0.001)
            price_tick = precision.get("price")  # Minimum price step (e.g., 0.01)
            min_amount = limits.get("amount", {}).get("min")  # Minimum order size in base currency
            min_cost = limits.get("cost", {}).get("min")  # Minimum order value in quote currency

            # --- Calculate Decimal Places using helper ---
            self.price_decimals = self._calculate_decimal_places(price_tick)
            self.amount_decimals = self._calculate_decimal_places(amount_tick)

            # Log warnings if essential info is missing or calculation failed
            if amount_tick is None:
                logger.warning(
                    f"{Fore.YELLOW}Amount precision (tick size) info missing for {self.symbol}. Using default amount decimals: {self.amount_decimals}. Amount formatting/checks may be inaccurate.{Style.RESET_ALL}"
                )
            if price_tick is None:
                logger.warning(
                    f"{Fore.YELLOW}Price precision (tick size) info missing for {self.symbol}. Using default price decimals: {self.price_decimals}. Price formatting/checks may be inaccurate.{Style.RESET_ALL}"
                )
            if min_amount is None:
                logger.warning(
                    f"{Fore.YELLOW}Minimum order amount limit missing for {self.symbol}. Cannot enforce minimum amount check effectively.{Style.RESET_ALL}"
                )
            if min_cost is None:
                logger.warning(
                    f"{Fore.YELLOW}Minimum order cost limit missing for {self.symbol}. Cannot enforce minimum cost check effectively.{Style.RESET_ALL}"
                )

            # Log key details including calculated decimals
            min_amount_str = f"{min_amount:.{self.amount_decimals}f}" if min_amount is not None else "N/A"
            min_cost_str = f"{min_cost:.{self.price_decimals}f}" if min_cost is not None else "N/A"
            logger.info(
                f"Market Details for {self.symbol}: "
                f"Price Tick={price_tick}, Price Decimals={self.price_decimals}, "
                f"Amount Tick={amount_tick}, Amount Decimals={self.amount_decimals}, "
                f"Min Amount={min_amount_str}, Min Cost={min_cost_str}"
            )
            logger.debug("Market details loaded and cached.")

        except KeyError as e:
            logger.critical(
                f"{Fore.RED}KeyError accessing market info for {self.symbol}: {e}. Market structure might be unexpected or incomplete. Market Info: {self.market_info}. Aborting.{Style.RESET_ALL}",
                exc_info=False,
            )
            sys.exit(1)
        except ValueError as e:  # Catch specific value errors from validation/retrieval
            logger.critical(
                f"{Fore.RED}ValueError loading market info for {self.symbol}: {e}. Aborting.{Style.RESET_ALL}",
                exc_info=False,
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}Failed to load or validate crucial market info for {self.symbol}: {e}. Cannot continue safely. Aborting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    # --- Data Fetching Methods ---

    @retry_api_call()
    def fetch_market_price(self) -> float | None:
        """Fetches the last traded price for the symbol."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        if ticker and ticker.get("last") is not None:
            try:
                price = float(ticker["last"])
                # Log with appropriate precision using stored decimal count
                logger.debug(f"Current market price ({self.symbol}): {price:.{self.price_decimals}f}")
                return price
            except (ValueError, TypeError) as e:
                logger.error(
                    f"{Fore.RED}Error converting ticker 'last' price to float for {self.symbol}. Value: {ticker.get('last')}. Error: {e}{Style.RESET_ALL}"
                )
                return None
        else:
            logger.warning(
                f"{Fore.YELLOW}Could not fetch valid 'last' price for {self.symbol}. Ticker data received: {ticker}{Style.RESET_ALL}"
            )
            return None

    @retry_api_call()
    def fetch_order_book(self) -> dict[str, Any] | None:
        """Fetches order book data (bids, asks) and calculates the volume imbalance."""
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        try:
            # params = {} # Add if needed for specific exchange behavior
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)  # , params=params)
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            result = {"bids": bids, "asks": asks, "imbalance": None}

            # Ensure bids/asks are lists of lists/tuples with at least price and volume
            # And volume is a valid number >= 0
            valid_bids = [
                bid
                for bid in bids
                if isinstance(bid, (list, tuple)) and len(bid) >= 2 and isinstance(bid[1], (int, float)) and bid[1] >= 0
            ]
            valid_asks = [
                ask
                for ask in asks
                if isinstance(ask, (list, tuple)) and len(ask) >= 2 and isinstance(ask[1], (int, float)) and ask[1] >= 0
            ]

            if not valid_bids or not valid_asks:
                logger.warning(
                    f"{Fore.YELLOW}Order book data invalid or incomplete for {self.symbol}. Valid Bids: {len(valid_bids)}, Valid Asks: {len(valid_asks)}. Raw: bids={len(bids)}, asks={len(asks)}{Style.RESET_ALL}"
                )
                # Return data structure but imbalance will be None
                return result

            # Calculate total volume within the fetched depth
            bid_volume = sum(float(bid[1]) for bid in valid_bids)
            ask_volume = sum(float(ask[1]) for ask in valid_asks)

            # Calculate imbalance ratio (Ask Volume / Bid Volume)
            # Handle potential division by zero and very small volumes
            epsilon = 1e-12  # Use a small epsilon to avoid floating point issues
            if bid_volume > epsilon:
                imbalance_ratio = ask_volume / bid_volume
                result["imbalance"] = imbalance_ratio
                logger.debug(f"Order Book ({self.symbol}) Imbalance (Ask Vol / Bid Vol): {imbalance_ratio:.3f}")
            elif ask_volume > epsilon:  # Bids are near zero, asks are not
                result["imbalance"] = float("inf")  # Represent infinite pressure upwards
                logger.debug(f"Order Book ({self.symbol}) Imbalance: Infinity (Near-Zero Bid Volume)")
            else:  # Both bid and ask volumes are effectively zero
                result["imbalance"] = None  # Undefined imbalance
                logger.debug(f"Order Book ({self.symbol}) Imbalance: N/A (Near-Zero Bid & Ask Volume)")

            return result
        except Exception as e:
            # Catch potential errors during fetch_order_book call itself
            logger.error(
                f"{Fore.RED}Error fetching or processing order book for {self.symbol}: {e}{Style.RESET_ALL}",
                exc_info=False,
            )  # Reduce log spam for common OB issues
            return None  # Indicate failure

    @retry_api_call(max_retries=2, initial_delay=1)  # Less critical, fewer retries
    def fetch_historical_data(self, limit: int | None = None) -> pd.DataFrame | None:
        """Fetches historical OHLCV data and prepares it as a Pandas DataFrame.
        Calculates the required number of candles based on indicator periods.

        Args:
            limit: Optional override for the number of candles to fetch.

        Returns:
            A pandas DataFrame with OHLCV data, or None on failure.
        """
        # Determine the minimum number of candles required for all indicators
        if limit is None:
            # Calculate max lookback needed by any indicator + buffer
            required_periods = [
                self.volatility_window,
                self.ema_period,
                self.rsi_period + 1,
                self.macd_long_period + self.macd_signal_period,  # Needs ~long+signal for good MACD hist
                self.stoch_rsi_period
                + max(self.stoch_rsi_k_period, self.stoch_rsi_d_period)
                + self.rsi_period,  # StochRSI needs RSI series first
                self.atr_period + 1,
            ]
            # Filter out zero or None periods before taking max
            valid_periods = [p for p in required_periods if p is not None and p > 0]
            max_lookback = max(valid_periods) if valid_periods else 0

            # Add a safety buffer (e.g., 25-50 candles)
            required_limit = max_lookback + 50
            # Ensure a minimum fetch limit if all periods are very small
            required_limit = max(required_limit, 50)
            # Cap the limit to avoid excessive requests (adjust as needed, check exchange limits)
            # CCXT might handle exchange limits automatically, but good to have a cap.
            required_limit = min(
                required_limit, 1000
            )  # Max limit depends on exchange (e.g., Bybit often 1000 or 1500 per request)
        else:
            required_limit = limit

        logger.debug(
            f"Fetching approximately {required_limit} historical candles for {self.symbol} ({self.timeframe})..."
        )
        try:
            # Fetch OHLCV data
            # params = {} # Add exchange-specific params if needed
            # Note: Some exchanges might return *fewer* than limit if history is short
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=required_limit
            )  # , params=params)

            if not ohlcv:
                logger.warning(
                    f"{Fore.YELLOW}No historical OHLCV data returned for {self.symbol} with timeframe {self.timeframe} and limit {required_limit}.{Style.RESET_ALL}"
                )
                return None
            if len(ohlcv) < 5:  # Arbitrary small number, need *some* data
                logger.warning(
                    f"{Fore.YELLOW}Very few historical OHLCV data points returned ({len(ohlcv)}). Insufficient for most indicators.{Style.RESET_ALL}"
                )
                return None

            # Convert to DataFrame and process
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)  # Ensure UTC timezone
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            initial_len = len(df)
            # Drop rows where essential price data is NaN
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            final_len = len(df)

            if final_len < initial_len:
                logger.debug(f"Dropped {initial_len - final_len} rows containing NaNs in OHLC from historical data.")

            # Check if enough data remains after cleaning for the longest indicator calculation
            # Reuse max_lookback calculated earlier if limit wasn't overridden
            if limit is None:
                min_required_len = max_lookback + 5  # Smaller buffer for check
            else:  # If limit was specified, use a reasonable minimum guess
                min_required_len = max(valid_periods) + 5 if valid_periods else 10

            if final_len < min_required_len:
                logger.warning(
                    f"{Fore.YELLOW}Insufficient valid historical data after cleaning for {self.symbol}. Fetched/Returned {initial_len}, valid {final_len}, needed ~{min_required_len} for full indicator calc. Indicators might be inaccurate or fail.{Style.RESET_ALL}"
                )
                if final_len == 0:
                    return None  # Definitely return None if empty

            logger.debug(f"Fetched and processed {final_len} valid historical candles for {self.symbol}.")
            return df

        except ccxt.BadSymbol as e:
            logger.critical(
                f"{Fore.RED}BadSymbol error fetching history for {self.symbol}. Is the symbol correct in config? Error: {e}{Style.RESET_ALL}"
            )
            sys.exit(1)  # Exit if symbol is fundamentally wrong
        except Exception as e:
            # Catch other potential errors during fetch or processing
            logger.error(
                f"{Fore.RED}Error fetching/processing historical data for {self.symbol}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            return None

    @retry_api_call()
    def fetch_balance(self, currency_code: str | None = None) -> float:
        """Fetches the 'free' or 'available' balance for a specific currency code (quote currency by default).
        Attempts to handle unified margin account structures (like Bybit V5).

        Args:
            currency_code: The currency code (e.g., 'USDT') to fetch the balance for. Defaults to the quote currency of the market.

        Returns:
            The available balance as a float, or 0.0 if unavailable or error occurs.
        """
        # Determine the quote currency from the market info if not provided
        quote_currency = currency_code or self.market_info.get("quote")
        if not quote_currency:
            logger.error(
                f"{Fore.RED}Cannot determine quote currency to fetch balance (market_info: {self.market_info}). Using 0.0{Style.RESET_ALL}"
            )
            return 0.0

        logger.debug(f"Fetching available balance for {quote_currency}...")

        # Handle simulation mode: return a dummy balance
        if self.simulation_mode:
            # Use price decimals for formatting the dummy balance for consistency
            dummy_balance = 10000.0
            logger.warning(
                f"{Fore.YELLOW}[SIMULATION] Returning dummy balance: {dummy_balance:.{self.price_decimals}f} {quote_currency}{Style.RESET_ALL}"
            )
            return dummy_balance

        try:
            # Add params if needed for specific account types (e.g., Bybit Unified vs Standard)
            # CCXT often handles this, but check if specific params are required
            params = {}
            # Example: Bybit might allow specifying accountType ('UNIFIED', 'CONTRACT', 'SPOT')
            # if self.exchange_id.lower() == 'bybit': params = {'accountType': 'UNIFIED'} # Or 'CONTRACT' / 'SPOT' if needed

            balance_data = self.exchange.fetch_balance(params=params)

            # --- Strategy: ---
            # 1. Try standard CCXT `balance[currency]['free']` first.
            # 2. If missing/zero, try parsing `balance['info']` which contains raw exchange data.
            # 3. If still missing/zero, try `balance['total']` as a fallback (less ideal, includes used margin).

            available_balance: str | float | None = None

            # --- 1. Standard 'free' balance ---
            if quote_currency in balance_data and balance_data[quote_currency].get("free") is not None:
                available_balance = balance_data[quote_currency]["free"]
                logger.debug(f"Found standard 'free' balance for {quote_currency}: {available_balance}")

            # --- 2. Parse 'info' (Exchange-Specific) ---
            # Check if 'free' was missing/zero, or if we specifically know 'info' is needed (e.g., Bybit Unified)
            # Use float conversion for check to handle "0.0" strings
            is_free_balance_zero = True
            try:
                if available_balance is not None and float(available_balance) > 1e-12:
                    is_free_balance_zero = False
            except (ValueError, TypeError):
                pass  # Treat conversion errors as zero/missing

            if is_free_balance_zero:
                logger.debug(
                    f"Standard 'free' balance for {quote_currency} is zero or missing. Checking 'info' for alternatives..."
                )
                info_data = balance_data.get("info", {})

                # --- Bybit V5 Specific Logic (Example) ---
                if self.exchange_id.lower() == "bybit" and isinstance(info_data, dict):
                    try:
                        # Bybit V5 /v5/account/wallet-balance structure
                        result_list = info_data.get("result", {}).get("list", [])
                        if result_list and isinstance(result_list[0], dict):
                            # Typically contains one entry for the account type (UNIFIED, CONTRACT, SPOT)
                            account_info = result_list[0]
                            account_type = account_info.get("accountType")  # e.g., 'UNIFIED', 'CONTRACT', 'SPOT'
                            logger.debug(f"Parsing Bybit 'info': Account Type = {account_type}")

                            # Unified/Contract Account (Margin Trading) - Prefer Available Balance
                            if account_type in ["UNIFIED", "CONTRACT"]:
                                # Bybit often provides 'totalAvailableBalance' for unified margin
                                unified_avail_str = account_info.get("totalAvailableBalance")
                                if unified_avail_str is not None:
                                    logger.debug(
                                        f"Using Bybit {account_type} 'totalAvailableBalance': {unified_avail_str}"
                                    )
                                    available_balance = unified_avail_str
                                else:
                                    # Fallback: Check coin-specific details within unified account
                                    coin_list = account_info.get("coin", [])
                                    for coin_entry in coin_list:
                                        if isinstance(coin_entry, dict) and coin_entry.get("coin") == quote_currency:
                                            # Prefer 'availableToBorrow' or 'availableToWithdraw' or 'walletBalance' ? Check Bybit docs.
                                            # 'availableBalance' in Bybit might be suitable for derivatives margin check.
                                            coin_avail = coin_entry.get(
                                                "availableBalance"
                                            )  # This seems most relevant for margin trading
                                            if coin_avail is None:
                                                coin_avail = coin_entry.get("availableToWithdraw")  # Fallback

                                            if coin_avail is not None:
                                                logger.debug(
                                                    f"Using coin-specific '{quote_currency}' balance ('availableBalance' or fallback) within {account_type} account: {coin_avail}"
                                                )
                                                available_balance = coin_avail
                                                break  # Found the specific coin balance

                            # SPOT Account Check
                            elif account_type == "SPOT":
                                coin_list = account_info.get("coin", [])
                                for coin_entry in coin_list:
                                    if isinstance(coin_entry, dict) and coin_entry.get("coin") == quote_currency:
                                        # Spot usually uses 'free' or equivalent. Check 'walletBalance'? 'free' might be availableToWithdraw.
                                        # Let's prioritize 'free' if present, else 'walletBalance'
                                        spot_avail = coin_entry.get("free")
                                        if spot_avail is None:
                                            spot_avail = coin_entry.get("walletBalance")  # Fallback

                                        if spot_avail is not None:
                                            logger.debug(
                                                f"Using Bybit SPOT balance ('free' or 'walletBalance') for {quote_currency}: {spot_avail}"
                                            )
                                            available_balance = spot_avail
                                            break
                        else:
                            logger.debug("Bybit 'info' structure not as expected (result.list missing or empty).")

                    except (AttributeError, IndexError, KeyError, TypeError) as bybit_info_err:
                        logger.warning(
                            f"{Fore.YELLOW}Could not parse expected Bybit structure in balance 'info': {bybit_info_err}. Raw info snippet: {str(info_data)[:200]}...{Style.RESET_ALL}"
                        )
                # --- Add logic for other exchanges' 'info' structure if needed ---
                # elif self.exchange_id.lower() == 'some_other_exchange':
                #     # Parse 'info' for that specific exchange
                #     pass

            # --- 3. Fallback to 'total' balance (Less Accurate) ---
            # Use float conversion for check
            is_available_balance_zero = True
            try:
                if available_balance is not None and float(available_balance) > 1e-12:
                    is_available_balance_zero = False
            except (ValueError, TypeError):
                pass

            if is_available_balance_zero and quote_currency in balance_data:
                total_balance = balance_data[quote_currency].get("total")
                if total_balance is not None:
                    logger.warning(
                        f"{Fore.YELLOW}Available balance is zero/missing. Using 'total' balance ({total_balance}) as a fallback for {quote_currency}. This includes used margin/collateral.{Style.RESET_ALL}"
                    )
                    available_balance = total_balance

            # --- Final Conversion and Return ---
            final_balance = 0.0
            if available_balance is not None:
                try:
                    final_balance = float(available_balance)
                    if final_balance < 0:
                        logger.warning(
                            f"{Fore.YELLOW}Fetched balance for {quote_currency} is negative ({final_balance}). Treating as 0.0.{Style.RESET_ALL}"
                        )
                        final_balance = 0.0
                except (ValueError, TypeError) as conv_err:
                    logger.error(
                        f"{Fore.RED}Could not convert final balance value '{available_balance}' to float for {quote_currency}: {conv_err}. Returning 0.0{Style.RESET_ALL}"
                    )
                    final_balance = 0.0
            else:
                logger.warning(
                    f"{Fore.YELLOW}Could not determine available balance for {quote_currency} after checking standard, info, and total fields. Returning 0.0.{Style.RESET_ALL}"
                )

            # Log balance with price precision for better readability of value
            logger.info(f"Fetched available balance: {final_balance:.{self.price_decimals}f} {quote_currency}")
            return final_balance

        except ccxt.AuthenticationError as e:
            logger.error(
                f"{Fore.RED}Authentication failed fetching balance for {quote_currency}: {e}. Ensure API keys are valid and have permissions.{Style.RESET_ALL}"
            )
            return 0.0  # Return 0 on auth error after init
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error fetching balance for {quote_currency}: {e}{Style.RESET_ALL}", exc_info=True
            )
            return 0.0

    @retry_api_call(max_retries=1)  # Don't retry 'fetch_order' aggressively if it fails initially
    def fetch_order_status(self, order_id: str, symbol: str | None = None) -> dict[str, Any] | None:
        """Fetches the status and details of a specific order by its ID.

        Args:
            order_id: The exchange's ID for the order.
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            The CCXT order dictionary or None if not found or error occurs.
            The dictionary may contain an 'info' key with raw exchange data.
        """
        if not order_id:
            logger.warning("fetch_order_status called with empty order_id.")
            return None

        target_symbol = symbol or self.symbol
        logger.debug(f"Fetching status for order {order_id} ({target_symbol})...")

        # Handle simulation mode: return placeholder based on internal state
        if self.simulation_mode:
            # Find the simulated order in our state
            for pos in self.open_positions:
                if pos["id"] == order_id:
                    logger.debug(f"[SIMULATION] Returning cached/simulated status for order {order_id}.")
                    # Construct a basic simulated order structure mimicking ccxt
                    sim_status = pos.get("status", STATUS_UNKNOWN)
                    # Translate internal status to potential ccxt status
                    ccxt_status = (
                        "open"
                        if sim_status == STATUS_PENDING_ENTRY
                        else "closed"
                        if sim_status == STATUS_ACTIVE
                        else "canceled"
                        if sim_status == STATUS_CANCELED
                        else sim_status
                    )  # Pass others like 'unknown'

                    sim_avg = pos.get("entry_price") if sim_status == STATUS_ACTIVE else None
                    sim_filled = pos.get("size") if sim_status == STATUS_ACTIVE else 0.0
                    sim_amount = pos.get("original_size", 0.0)

                    entry_time_sec = pos.get("entry_time")
                    sim_timestamp_ms = int(entry_time_sec * 1000) if entry_time_sec else int(time.time() * 1000)
                    sim_datetime = pd.to_datetime(sim_timestamp_ms, unit="ms", utc=True).isoformat()

                    return {
                        "id": order_id,
                        "clientOrderId": order_id,
                        "symbol": target_symbol,
                        "status": ccxt_status,
                        "type": pos.get("entry_order_type", "limit"),
                        "side": pos.get("side"),
                        "amount": sim_amount,
                        "filled": sim_filled,
                        "remaining": max(0, sim_amount - sim_filled),
                        "average": sim_avg,
                        "timestamp": sim_timestamp_ms,
                        "datetime": sim_datetime,
                        "stopLossPrice": pos.get("stop_loss_price"),  # Reflects internal state SL/TP
                        "takeProfitPrice": pos.get("take_profit_price"),
                        "info": {"simulated": True, "orderId": order_id, "internalStatus": sim_status},
                    }
            logger.warning(f"[SIMULATION] Simulated order {order_id} not found in current state for status check.")
            return None  # Not found in simulation state

        # --- Live Mode ---
        else:
            try:
                # Add params if needed (e.g., {'category': 'linear'} for Bybit V5 derivatives)
                params = {}
                if self.exchange_id.lower() == "bybit":
                    market_type = self.exchange.market(target_symbol).get("type", "swap")
                    if market_type in ["swap", "future"]:
                        params["category"] = (
                            "linear" if self.exchange.market(target_symbol).get("linear") else "inverse"
                        )
                    elif market_type == "spot":
                        # Spot order history might not need category or might use 'spot'
                        params["category"] = "spot"  # Example, check Bybit API if needed

                order_info = self.exchange.fetch_order(order_id, target_symbol, params=params)

                # Log key details using decimals formatting
                status = order_info.get("status", STATUS_UNKNOWN)
                amount = order_info.get("amount", 0.0)
                filled = order_info.get("filled", 0.0)
                avg_price = order_info.get("average")
                remaining = order_info.get("remaining")
                order_type = order_info.get("type")
                order_side = order_info.get("side")
                amount_str = f"{amount:.{self.amount_decimals}f}" if amount is not None else "N/A"
                filled_str = f"{filled:.{self.amount_decimals}f}" if filled is not None else "N/A"
                avg_price_str = f"{avg_price:.{self.price_decimals}f}" if avg_price is not None else "N/A"
                remaining_str = f"{remaining:.{self.amount_decimals}f}" if remaining is not None else "N/A"

                logger.debug(
                    f"Order {order_id} ({order_type} {order_side}): Status={status}, Amount={amount_str}, Filled={filled_str}, AvgPrice={avg_price_str}, Remaining={remaining_str}"
                )

                return order_info

            except ccxt.OrderNotFound as e:
                # This is common, treat as non-error, indicates order is final or never existed
                logger.warning(
                    f"{Fore.YELLOW}Order {order_id} not found on exchange ({target_symbol}). Assumed closed, cancelled, or invalid ID. Error: {e}{Style.RESET_ALL}"
                )
                return None  # Return None, let the calling function decide how to interpret this
            except ccxt.ExchangeError as e:
                # Check if it's a potentially recoverable error string before logging as ERROR
                err_str = str(e).lower()
                if (
                    "order is finished" in err_str or "order has been filled" in err_str or "already closed" in err_str
                ):  # Common variations for closed orders
                    logger.warning(
                        f"{Fore.YELLOW}Order {order_id} not found (reported as finished/filled/closed). Assuming closed/cancelled. Error: {e}{Style.RESET_ALL}"
                    )
                    return None
                else:
                    # Log other exchange errors, but usually don't retry fetch_order unless network error
                    logger.error(
                        f"{Fore.RED}Exchange error fetching status for order {order_id}: {e}. Returning None.{Style.RESET_ALL}"
                    )
                    return None  # Let retry handle network, otherwise fail here
            except Exception as e:
                # Catch unexpected errors
                logger.error(
                    f"{Fore.RED}Unexpected error fetching status for order {order_id}: {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                return None

    # --- Indicator Calculation Methods ---
    # These methods are static as they operate purely on input data
    # Added basic error handling and checks for sufficient data length

    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> float | None:
        """Calculates the rolling standard deviation of log returns."""
        if close_prices is None or window <= 0:
            return None
        min_len = window + 1  # Need window + 1 prices for 'window' returns
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for volatility (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            # Ensure prices are positive for log calculation
            if (close_prices <= 0).any():
                logger.warning("Non-positive prices found, cannot calculate log returns for volatility.")
                return None
            log_returns = np.log(close_prices / close_prices.shift(1))
            # Use pandas std() which calculates sample std dev by default (ddof=1)
            volatility = log_returns.rolling(window=window, min_periods=window).std(ddof=1).iloc[-1]
            return float(volatility) if pd.notna(volatility) else None
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}", exc_info=False)  # Avoid spamming logs
            return None

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> float | None:
        """Calculates the Exponential Moving Average (EMA)."""
        if close_prices is None or period <= 0:
            return None
        # EMA needs 'period' points for decent initialization with adjust=False
        min_len = period
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for EMA {period} (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            # adjust=False uses the recursive definition: EMA_today = alpha * Price_today + (1 - alpha) * EMA_yesterday
            ema = close_prices.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            return float(ema) if pd.notna(ema) else None
        except Exception as e:
            logger.error(f"Error calculating EMA {period}: {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> float | None:
        """Calculates the Relative Strength Index (RSI)."""
        if close_prices is None or period <= 0:
            return None
        min_len = period + 1  # Needs at least period+1 for the initial diff
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for RSI {period} (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            # Use Exponential Moving Average for smoothing gains and losses (common method)
            # Ensure min_periods matches the period for robust calculation start
            avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

            # Calculate Relative Strength (RS)
            # Add epsilon to avoid division by zero if avg_loss is zero
            rs = avg_gain / (avg_loss + 1e-12)

            # Calculate RSI
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_val = rsi.iloc[-1]
            # Clamp RSI between 0 and 100 (can sometimes slightly exceed due to floating point)
            rsi_val = max(0.0, min(100.0, rsi_val)) if pd.notna(rsi_val) else None
            return float(rsi_val) if rsi_val is not None else None
        except Exception as e:
            logger.error(f"Error calculating RSI {period}: {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_macd(
        close_prices: pd.Series, short_period: int, long_period: int, signal_period: int
    ) -> tuple[float | None, float | None, float | None]:
        """Calculates the Moving Average Convergence Divergence (MACD), Signal Line, and Histogram."""
        if close_prices is None or not all(p > 0 for p in [short_period, long_period, signal_period]):
            return None, None, None
        if short_period >= long_period:
            logger.error(f"MACD short_period ({short_period}) must be less than long_period ({long_period}).")
            return None, None, None
        # Rough minimum length for meaningful values (long EMA needs init, signal EMA needs MACD series)
        min_len = long_period + signal_period
        if len(close_prices) < min_len:
            logger.debug(
                f"Insufficient data for MACD ({short_period},{long_period},{signal_period}) (need ~{min_len}, got {len(close_prices)})."
            )
            return None, None, None
        try:
            ema_short = close_prices.ewm(span=short_period, adjust=False, min_periods=short_period).mean()
            ema_long = close_prices.ewm(span=long_period, adjust=False, min_periods=long_period).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
            histogram = macd_line - signal_line

            # Get the latest values
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]
            hist_val = histogram.iloc[-1]

            # Return None if any value is NaN (e.g., due to insufficient data at the end)
            if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
                logger.debug("MACD calculation resulted in NaN (likely insufficient data for period).")
                return None, None, None
            return float(macd_val), float(signal_val), float(hist_val)
        except Exception as e:
            logger.error(f"Error calculating MACD ({short_period},{long_period},{signal_period}): {e}", exc_info=False)
            return None, None, None

    @staticmethod
    def calculate_stoch_rsi(
        close_prices: pd.Series, rsi_period: int, stoch_period: int, k_period: int, d_period: int
    ) -> tuple[float | None, float | None]:
        """Calculates the Stochastic RSI (%K and %D)."""
        if close_prices is None or not all(p > 0 for p in [rsi_period, stoch_period, k_period, d_period]):
            return None, None
        # Estimate minimum length: Need RSI series, then rolling Stoch window, then smoothing
        min_len = rsi_period + stoch_period + max(k_period, d_period)
        if len(close_prices) < min_len:
            logger.debug(
                f"Insufficient data for StochRSI ({rsi_period},{stoch_period},{k_period},{d_period}) (need ~{min_len}, got {len(close_prices)})."
            )
            return None, None
        try:
            # --- Calculate RSI Series first ---
            rsi_series = ScalpingBot.calculate_rsi_series(close_prices, rsi_period)
            if rsi_series is None or rsi_series.isna().all():
                logger.debug("RSI series calculation failed or returned all NaNs for StochRSI.")
                return None, None

            rsi_series = rsi_series.dropna()  # Drop initial NaNs from RSI calc
            if len(rsi_series) < stoch_period:
                logger.debug(
                    f"Insufficient non-NaN RSI series length for StochRSI window (need {stoch_period}, got {len(rsi_series)})."
                )
                return None, None

            # --- Calculate Stochastic of RSI ---
            min_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min()
            max_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).max()
            # Add epsilon to denominator to avoid division by zero if max == min over the window
            stoch_rsi_raw = (100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-12)).clip(0, 100)

            # --- Calculate %K and %D (SMA smoothing) ---
            stoch_k = stoch_rsi_raw.rolling(window=k_period, min_periods=k_period).mean()
            stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

            k_val, d_val = stoch_k.iloc[-1], stoch_d.iloc[-1]

            if pd.isna(k_val) or pd.isna(d_val):
                logger.debug("StochRSI K or D is NaN (likely insufficient smoothed data).")
                return None, None
            # Clamp final values just in case
            k_val = max(0.0, min(100.0, k_val))
            d_val = max(0.0, min(100.0, d_val))
            return float(k_val), float(d_val)
        except Exception as e:
            logger.error(
                f"Error calculating StochRSI ({rsi_period},{stoch_period},{k_period},{d_period}): {e}", exc_info=False
            )
            return None, None

    @staticmethod
    def calculate_rsi_series(close_prices: pd.Series, period: int) -> pd.Series | None:
        """Helper to calculate the full RSI series needed for StochRSI."""
        if close_prices is None or period <= 0:
            return None
        min_len = period + 1
        if len(close_prices) < min_len:
            return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-12)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi
        except Exception:
            return None  # Return None on any calculation error

    @staticmethod
    def calculate_atr(
        high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int
    ) -> float | None:
        """Calculates the Average True Range (ATR)."""
        if high_prices is None or low_prices is None or close_prices is None or period <= 0:
            return None
        min_len = period + 1  # Needs prior close for TR calculation
        if not (len(high_prices) >= min_len and len(low_prices) >= min_len and len(close_prices) >= min_len):
            logger.debug(f"Insufficient data for ATR {period} (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))

            # Combine the three components to find the True Range (TR)
            # Ensure we use .values to work with numpy arrays for max operation if needed
            tr_df = pd.DataFrame({"hl": high_low, "hc": high_close, "lc": low_close})
            tr = tr_df.max(axis=1, skipna=False)  # skipna=False propagates NaN if any component is NaN
            # First TR value will be NaN due to shift(1), handle it
            tr = tr.fillna(
                0
            )  # Fill initial NaN with 0 (or handle differently if desired, e.g., start ATR after first TR)

            # Calculate ATR using Exponential Moving Average (common method)
            # Ensure min_periods matches period for robust start
            atr = tr.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            # Alternative using Wilder's Smoothing (alpha = 1/period => com = period - 1):
            # atr = tr.ewm(com=period-1, adjust=False, min_periods=period).mean().iloc[-1]

            return float(atr) if pd.notna(atr) and atr >= 0 else None  # Ensure ATR is non-negative
        except Exception as e:
            logger.error(f"Error calculating ATR {period}: {e}", exc_info=False)
            return None

    # --- Trading Logic & Order Management Methods ---

    def calculate_order_size(self, current_price: float) -> float:
        """Calculates the order size in the BASE currency based on available balance,
        percentage risk, market limits, and optional volatility adjustment.
        Crucially checks against minimum amount and cost limits AFTER precision.

        Args:
            current_price: The current market price.

        Returns:
            The calculated order size in BASE currency (e.g., DOT), rounded to
            exchange precision, or 0.0 if checks fail or size is below minimums.
        """
        if self.market_info is None:
            logger.error("Market info not loaded, cannot calculate order size.")
            return 0.0
        if current_price <= 1e-12:  # Avoid division by zero / nonsensical prices
            logger.error(f"Current price ({current_price}) is zero or negative, cannot calculate order size.")
            return 0.0

        quote_currency = self.market_info["quote"]
        base_currency = self.market_info["base"]
        min_amount = self.market_info.get("limits", {}).get("amount", {}).get("min")
        min_cost = self.market_info.get("limits", {}).get("cost", {}).get("min")
        amount_tick_size = self.market_info.get("precision", {}).get("amount")  # The smallest increment step

        # --- 1. Get Available Balance ---
        balance = self.fetch_balance(currency_code=quote_currency)
        if balance <= 0:
            # This is expected if funds run out, log as INFO or WARNING
            logger.info(
                f"Insufficient balance ({balance:.{self.price_decimals}f} {quote_currency}) for new order. Need more {quote_currency}."
            )
            return 0.0

        # --- 2. Calculate Base Order Size (Quote Value) ---
        base_order_size_quote = balance * self.order_size_percentage
        logger.debug(
            f"Initial calculated order size (quote): {base_order_size_quote:.{self.price_decimals}f} {quote_currency} (Balance: {balance:.{self.price_decimals}f}, Pct: {self.order_size_percentage * 100:.2f}%)"
        )
        if base_order_size_quote <= 0:
            logger.warning("Calculated base order size (quote) is zero or negative. Cannot proceed.")
            return 0.0

        # --- 3. Optional Volatility Adjustment ---
        final_size_quote = base_order_size_quote
        if self.volatility_multiplier is not None and self.volatility_multiplier > 0:
            logger.debug("Attempting volatility adjustment for order size...")
            # Requires fetching history again if indicators aren't passed; consider passing them.
            # For now, re-fetch limited history.
            hist_data_vol = self.fetch_historical_data(limit=self.volatility_window + 5)
            volatility = None
            if hist_data_vol is not None and not hist_data_vol.empty:
                volatility = self.calculate_volatility(hist_data_vol["close"], self.volatility_window)

            if volatility is not None and volatility > 1e-9:
                # Inverse relationship: higher volatility -> smaller size factor
                # Scaling factor needs careful tuning based on typical volatility values
                # Example: size_factor = 1 / (1 + volatility * multiplier_scale)
                # Assuming volatility is stdev of log returns (e.g., 0.01 = 1%), use appropriate scale.
                # Let's keep the original simple scaling for now, assuming multiplier takes scale into account.
                size_factor = 1 / (1 + volatility * self.volatility_multiplier)  # Simplified inverse scaling
                size_factor = max(0.1, min(2.0, size_factor))  # Clamp adjustment factor (e.g., 0.1x to 2.0x)
                final_size_quote = base_order_size_quote * size_factor
                logger.info(
                    f"Volatility ({volatility:.5f}) adjustment factor: {size_factor:.3f}. "
                    f"Adjusted size (quote): {final_size_quote:.{self.price_decimals}f} {quote_currency} (Base: {base_order_size_quote:.{self.price_decimals}f})"
                )
            else:
                logger.debug(
                    f"Volatility adjustment skipped: Volatility N/A ({volatility}) or multiplier is zero/invalid."
                )
        else:
            logger.debug("Volatility adjustment disabled or multiplier is zero.")

        # --- 4. Convert Quote Size to Base Amount ---
        if current_price <= 1e-12:  # Re-check after potential adjustments
            logger.error("Current price is zero/negative after adjustments. Cannot convert to base amount.")
            return 0.0
        order_size_base_raw = final_size_quote / current_price
        logger.debug(
            f"Calculated raw base amount: {order_size_base_raw:.{self.amount_decimals + 4}f} {base_currency}"
        )  # Log with extra precision before rounding

        # --- 5. Apply Exchange Precision AND Check Limits ---
        # This is the critical section addressing the log errors.
        try:
            # Apply Amount Precision using CCXT helper
            # This rounds the amount according to the exchange rules (e.g., to amount_tick_size)
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base_raw)
            amount_precise = float(amount_precise_str)
            logger.debug(
                f"Amount after applying exchange precision: {amount_precise:.{self.amount_decimals}f} {base_currency}"
            )

            # Check if precision resulted in zero amount
            if amount_precise <= 1e-12:  # Use epsilon
                logger.warning(
                    f"{Fore.YELLOW}Order amount became zero after applying precision ({amount_tick_size}). Original raw amount: {order_size_base_raw:.{self.amount_decimals + 4}f}. Cannot place order.{Style.RESET_ALL}"
                )
                return 0.0

            # Apply Price Precision (needed for accurate cost calculation)
            price_precise_str = self.exchange.price_to_precision(self.symbol, current_price)
            price_precise = float(price_precise_str)

            # Check Minimum Amount Limit (Base Currency)
            if min_amount is not None and amount_precise < min_amount:
                # This is a likely cause of the original error log.
                logger.warning(
                    f"{Fore.YELLOW}Calculated precise order amount {amount_precise:.{self.amount_decimals}f} {base_currency} is below exchange minimum required: {min_amount}. Cannot place order.{Style.RESET_ALL}"
                )
                return 0.0

            # Check Minimum Cost Limit (Quote Currency)
            estimated_cost = amount_precise * price_precise
            if min_cost is not None and estimated_cost < min_cost:
                logger.warning(
                    f"{Fore.YELLOW}Estimated order cost {estimated_cost:.{self.price_decimals}f} {quote_currency} is below exchange minimum required: {min_cost}. Cannot place order. (Amt: {amount_precise}, Price: {price_precise}){Style.RESET_ALL}"
                )
                return 0.0

            # Log success if all checks passed
            logger.info(
                f"{Fore.CYAN}Calculated final order size: {amount_precise:.{self.amount_decimals}f} {base_currency} "
                f"(Est. Cost: {estimated_cost:.{self.price_decimals}f} {quote_currency}){Style.RESET_ALL}"
            )

            return amount_precise

        except ccxt.InvalidOrder as e:
            # Catch specific errors from amount_to_precision if it flags minimums directly
            # This complements the explicit checks above.
            logger.error(
                f"{Fore.RED}Error applying precision or checking limits (ccxt.InvalidOrder): {e}. Raw amount: {order_size_base_raw:.{self.amount_decimals + 4}f}. Cannot place order.{Style.RESET_ALL}"
            )
            return 0.0
        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            # Catch other potential errors during precision/limit handling
            logger.error(
                f"{Fore.RED}Error applying precision/limits: {type(e).__name__} - {e}. Raw amount: {order_size_base_raw:.{self.amount_decimals + 4}f}.{Style.RESET_ALL}"
            )
            return 0.0
        except Exception as e:  # Catch any other calculation error
            logger.error(
                f"{Fore.RED}Unexpected error during order size precision/limit checks: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            return 0.0

    def _calculate_sl_tp_prices(
        self, entry_price: float, side: str, current_price: float, atr: float | None
    ) -> tuple[float | None, float | None]:
        """Calculates Stop Loss (SL) and Take Profit (TP) prices based on configuration.
        Uses either ATR or fixed percentages from the entry price.
        Includes sanity checks against current price and entry price, and applies exchange precision.

        Args:
            entry_price: The estimated or actual entry price of the position.
            side: 'buy' (long) or 'sell' (short).
            current_price: The current market price, used for sanity checks.
            atr: The current ATR value (required if use_atr_sl_tp is True).

        Returns:
            A tuple containing (stop_loss_price, take_profit_price) as floats,
            correctly formatted to the exchange's price precision, or None if calculation fails.
        """
        stop_loss_price_raw: float | None = None
        take_profit_price_raw: float | None = None
        stop_loss_price_final: float | None = None
        take_profit_price_final: float | None = None

        if self.market_info is None or entry_price <= 1e-12:
            logger.error(f"Market info missing or invalid entry price ({entry_price}) for SL/TP calc.")
            return None, None

        # Helper to format price for logging using stored decimals
        def format_price(p: float | None) -> str:
            if p is None:
                return "N/A"
            try:
                return f"{p:.{self.price_decimals}f}"
            except (TypeError, ValueError):
                return str(p)  # Fallback if formatting fails

        # --- Calculate Raw SL/TP based on Method ---
        if self.use_atr_sl_tp:
            if atr is None or atr <= 1e-12:
                logger.warning(
                    f"{Fore.YELLOW}ATR SL/TP enabled, but ATR is invalid ({atr}). Cannot calculate SL/TP.{Style.RESET_ALL}"
                )
                return None, None
            if not (
                self.atr_sl_multiplier
                and self.atr_tp_multiplier
                and self.atr_sl_multiplier > 0
                and self.atr_tp_multiplier > 0
            ):
                logger.warning(
                    f"{Fore.YELLOW}ATR SL/TP enabled, but multipliers invalid (SL={self.atr_sl_multiplier}, TP={self.atr_tp_multiplier}). Cannot calculate SL/TP.{Style.RESET_ALL}"
                )
                return None, None

            logger.debug(
                f"Calculating SL/TP using ATR={format_price(atr)}, SL Mult={self.atr_sl_multiplier}, TP Mult={self.atr_tp_multiplier}"
            )
            sl_delta = atr * self.atr_sl_multiplier
            tp_delta = atr * self.atr_tp_multiplier

            if side == "buy":
                stop_loss_price_raw = entry_price - sl_delta
                take_profit_price_raw = entry_price + tp_delta
            elif side == "sell":
                stop_loss_price_raw = entry_price + sl_delta
                take_profit_price_raw = entry_price - tp_delta
            else:
                logger.error(f"Invalid side '{side}' for SL/TP calculation.")
                return None, None

        else:  # Use Fixed Percentage
            sl_pct = self.base_stop_loss_pct
            tp_pct = self.base_take_profit_pct
            if not (sl_pct and tp_pct and 0 < sl_pct < 1 and 0 < tp_pct < 1):
                logger.warning(
                    f"{Fore.YELLOW}Fixed % SL/TP enabled, but percentages invalid/missing (SL={sl_pct}, TP={tp_pct}). Cannot calculate SL/TP.{Style.RESET_ALL}"
                )
                return None, None

            logger.debug(f"Calculating SL/TP using Fixed %: SL={sl_pct * 100:.2f}%, TP={tp_pct * 100:.2f}%")
            if side == "buy":
                stop_loss_price_raw = entry_price * (1 - sl_pct)
                take_profit_price_raw = entry_price * (1 + tp_pct)
            elif side == "sell":
                stop_loss_price_raw = entry_price * (1 + sl_pct)
                take_profit_price_raw = entry_price * (1 - tp_pct)
            else:
                logger.error(f"Invalid side '{side}' for SL/TP calculation.")
                return None, None

        # --- Apply Precision and Initial Sanity Checks ---
        try:
            # Process Stop Loss
            if stop_loss_price_raw is not None and stop_loss_price_raw > 1e-12:
                stop_loss_price_final = float(self.exchange.price_to_precision(self.symbol, stop_loss_price_raw))
                # Check if SL is now illogical relative to entry after precision
                if (side == "buy" and stop_loss_price_final >= entry_price) or (
                    side == "sell" and stop_loss_price_final <= entry_price
                ):
                    logger.warning(
                        f"{Fore.YELLOW}Calculated SL {format_price(stop_loss_price_final)} is on wrong side of entry {format_price(entry_price)} after precision. Setting SL to None.{Style.RESET_ALL}"
                    )
                    stop_loss_price_final = None
            elif stop_loss_price_raw is not None:  # If raw SL was zero or negative
                logger.warning(
                    f"Raw calculated stop loss price ({format_price(stop_loss_price_raw)}) is zero or negative. Setting SL to None."
                )
                stop_loss_price_final = None

            # Process Take Profit
            if take_profit_price_raw is not None and take_profit_price_raw > 1e-12:
                take_profit_price_final = float(self.exchange.price_to_precision(self.symbol, take_profit_price_raw))
                # Check if TP is now illogical relative to entry after precision
                if (side == "buy" and take_profit_price_final <= entry_price) or (
                    side == "sell" and take_profit_price_final >= entry_price
                ):
                    logger.warning(
                        f"{Fore.YELLOW}Calculated TP {format_price(take_profit_price_final)} is on wrong side of entry {format_price(entry_price)} after precision. Setting TP to None.{Style.RESET_ALL}"
                    )
                    take_profit_price_final = None
            elif take_profit_price_raw is not None:  # If raw TP was zero or negative
                logger.warning(
                    f"Raw calculated take profit price ({format_price(take_profit_price_raw)}) is zero or negative. Setting TP to None."
                )
                take_profit_price_final = None

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(
                f"{Fore.RED}Error applying precision to SL/TP prices: {e}. Raw SL={format_price(stop_loss_price_raw)}, TP={format_price(take_profit_price_raw)}. Setting both to None.{Style.RESET_ALL}"
            )
            return None, None
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error applying precision to SL/TP: {e}. Setting both to None.{Style.RESET_ALL}",
                exc_info=True,
            )
            return None, None

        # --- Final Sanity Check: SL vs TP ---
        if stop_loss_price_final is not None and take_profit_price_final is not None:
            if (side == "buy" and stop_loss_price_final >= take_profit_price_final) or (
                side == "sell" and stop_loss_price_final <= take_profit_price_final
            ):
                logger.warning(
                    f"{Fore.YELLOW}Final check: Calculated SL {format_price(stop_loss_price_final)} conflicts with TP {format_price(take_profit_price_final)} for {side.upper()} position. Setting TP to None to avoid conflict.{Style.RESET_ALL}"
                )
                take_profit_price_final = None

        # --- Final Sanity Check: SL/TP vs Current Price (Avoid Immediate Trigger?) ---
        # This check is debatable. Parameter-based SL/TP might handle this better.
        # If enabled, it might prevent placing orders if the market moves quickly.
        # Keeping it simple for now, relying on the exchange to handle SL/TP placement logic.
        # Could add a check like: if abs(current_price - sl_price) < min_price_increment * N ... warn/adjust
        # price_tick = float(self.market_info['precision']['price'] or 0.0001)
        # buffer = price_tick * 5 # Example: 5 ticks buffer
        # if stop_loss_price_final is not None:
        #      if (side == "buy" and stop_loss_price_final >= current_price - buffer) or \
        #         (side == "sell" and stop_loss_price_final <= current_price + buffer):
        #            logger.warning(f"{Fore.YELLOW}Calculated SL {format_price(stop_loss_price_final)} is very close to current price {format_price(current_price)}. Potential for immediate trigger.{Style.RESET_ALL}")
        #            # Optionally adjust or set to None here if this is problematic
        # if take_profit_price_final is not None:
        #      # Similar check for TP
        #      pass

        logger.debug(
            f"Final calculated SL={format_price(stop_loss_price_final)}, TP={format_price(take_profit_price_final)} for {side} entry near {format_price(entry_price)}"
        )
        return stop_loss_price_final, take_profit_price_final

    def compute_trade_signal_score(
        self,
        price: float,
        indicators: dict[str, float | tuple[float | None, ...] | None],
        orderbook_imbalance: float | None,
    ) -> tuple[int, list[str]]:
        """Computes a simple trade signal score based on configured indicators and order book imbalance.

        Score Contributions (Example - adjust weights/logic as needed):
        - Order Book Imbalance: +1.0 (Buy Pressure), -1.0 (Sell Pressure)
        - Price vs EMA:        +1.0 (Above), -1.0 (Below)
        - RSI:                 +1.0 (Oversold < 35), -1.0 (Overbought > 65)
        - MACD:                +1.0 (Line > Signal), -1.0 (Line <= Signal)
        - StochRSI:            +1.0 (Both K/D < 25), -1.0 (Both K/D > 75)

        Args:
            price: Current market price.
            indicators: Dictionary of calculated indicator values.
            orderbook_imbalance: Calculated order book imbalance ratio (Ask Vol / Bid Vol), can be None or inf.

        Returns:
            A tuple containing the final integer score and a list of string reasons for the score components.
        """
        score = 0.0
        reasons = []
        # Define thresholds (consider making these configurable)
        RSI_OVERSOLD, RSI_OVERBOUGHT = 35, 65
        STOCH_OVERSOLD, STOCH_OVERBOUGHT = 25, 75
        EMA_THRESHOLD_MULTIPLIER = 0.0002  # Price must be 0.02% above/below EMA to trigger

        # --- 1. Order Book Imbalance ---
        if orderbook_imbalance is not None:
            if self.imbalance_threshold <= 0:
                reason_str = f"[ 0.0] OB Invalid Threshold ({self.imbalance_threshold})"
            elif orderbook_imbalance == float("inf"):  # Handle infinite imbalance (zero bids)
                score -= 1.0  # Strong sell pressure if asks exist but bids are zero
                reason_str = (
                    f"{Fore.RED}[-1.0] OB Sell Pressure (Imb: Inf > {self.imbalance_threshold:.2f}){Style.RESET_ALL}"
                )
            else:  # Finite imbalance ratio
                imb_buy_thresh = 1.0 / self.imbalance_threshold if self.imbalance_threshold > 0 else float("inf")
                imb = orderbook_imbalance
                if imb < imb_buy_thresh:
                    score += 1.0
                    reason_str = (
                        f"{Fore.GREEN}[+1.0] OB Buy Pressure (Imb: {imb:.2f} < {imb_buy_thresh:.2f}){Style.RESET_ALL}"
                    )
                elif imb > self.imbalance_threshold:
                    score -= 1.0
                    reason_str = f"{Fore.RED}[-1.0] OB Sell Pressure (Imb: {imb:.2f} > {self.imbalance_threshold:.2f}){Style.RESET_ALL}"
                else:  # Between thresholds
                    reason_str = f"{Fore.WHITE}[ 0.0] OB Balanced ({imb_buy_thresh:.2f} <= Imb: {imb:.2f} <= {self.imbalance_threshold:.2f}){Style.RESET_ALL}"
        else:
            reason_str = f"{Fore.WHITE}[ 0.0] OB Data N/A{Style.RESET_ALL}"
        reasons.append(reason_str)

        # --- 2. EMA Trend ---
        ema = indicators.get("ema")
        if ema is not None and ema > 1e-9:
            ema_upper_bound = ema * (1 + EMA_THRESHOLD_MULTIPLIER)
            ema_lower_bound = ema * (1 - EMA_THRESHOLD_MULTIPLIER)
            price_f = f"{price:.{self.price_decimals}f}"
            ema_f = f"{ema:.{self.price_decimals}f}"
            if price > ema_upper_bound:
                score += 1.0
                reason_str = f"{Fore.GREEN}[+1.0] Price > EMA ({price_f} > {ema_f}){Style.RESET_ALL}"
            elif price < ema_lower_bound:
                score -= 1.0
                reason_str = f"{Fore.RED}[-1.0] Price < EMA ({price_f} < {ema_f}){Style.RESET_ALL}"
            else:
                reason_str = f"{Fore.WHITE}[ 0.0] Price near EMA ({price_f} ~ {ema_f}){Style.RESET_ALL}"
        else:
            reason_str = f"{Fore.WHITE}[ 0.0] EMA N/A{Style.RESET_ALL}"
        reasons.append(reason_str)

        # --- 3. RSI Momentum/OB/OS ---
        rsi = indicators.get("rsi")
        if rsi is not None:
            rsi_f = f"{rsi:.1f}"
            if rsi < RSI_OVERSOLD:
                score += 1.0
                reason_str = f"{Fore.GREEN}[+1.0] RSI Oversold ({rsi_f} < {RSI_OVERSOLD}){Style.RESET_ALL}"
            elif rsi > RSI_OVERBOUGHT:
                score -= 1.0
                reason_str = f"{Fore.RED}[-1.0] RSI Overbought ({rsi_f} > {RSI_OVERBOUGHT}){Style.RESET_ALL}"
            else:
                reason_str = (
                    f"{Fore.WHITE}[ 0.0] RSI Neutral ({RSI_OVERSOLD} <= {rsi_f} <= {RSI_OVERBOUGHT}){Style.RESET_ALL}"
                )
        else:
            reason_str = f"{Fore.WHITE}[ 0.0] RSI N/A{Style.RESET_ALL}"
        reasons.append(reason_str)

        # --- 4. MACD Momentum/Cross ---
        macd_line, macd_signal = indicators.get("macd_line"), indicators.get("macd_signal")
        if macd_line is not None and macd_signal is not None:
            # Format using more decimals for comparison if needed
            macd_f = f"{macd_line:.{self.price_decimals + 1}f}"
            sig_f = f"{macd_signal:.{self.price_decimals + 1}f}"
            if macd_line > macd_signal:
                score += 1.0
                reason_str = f"{Fore.GREEN}[+1.0] MACD Line > Signal ({macd_f} > {sig_f}){Style.RESET_ALL}"
            else:  # MACD <= Signal
                score -= 1.0
                reason_str = f"{Fore.RED}[-1.0] MACD Line <= Signal ({macd_f} <= {sig_f}){Style.RESET_ALL}"
        else:
            reason_str = f"{Fore.WHITE}[ 0.0] MACD N/A{Style.RESET_ALL}"
        reasons.append(reason_str)

        # --- 5. Stochastic RSI OB/OS ---
        stoch_k, stoch_d = indicators.get("stoch_k"), indicators.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            k_f, d_f = f"{stoch_k:.1f}", f"{stoch_d:.1f}"
            # Condition for Oversold: Both K and D below threshold
            if stoch_k < STOCH_OVERSOLD and stoch_d < STOCH_OVERSOLD:
                score += 1.0
                reason_str = (
                    f"{Fore.GREEN}[+1.0] StochRSI Oversold (K={k_f}, D={d_f} < {STOCH_OVERSOLD}){Style.RESET_ALL}"
                )
            # Condition for Overbought: Both K and D above threshold
            elif stoch_k > STOCH_OVERBOUGHT and stoch_d > STOCH_OVERBOUGHT:
                score -= 1.0
                reason_str = (
                    f"{Fore.RED}[-1.0] StochRSI Overbought (K={k_f}, D={d_f} > {STOCH_OVERBOUGHT}){Style.RESET_ALL}"
                )
            # Optional: Add crossover signal check (requires previous K/D)
            # elif prev_k < prev_d and k >= d and k < 80: score += 0.5 ...
            else:
                reason_str = f"{Fore.WHITE}[ 0.0] StochRSI Neutral/Cross (K={k_f}, D={d_f}){Style.RESET_ALL}"
        else:
            reason_str = f"{Fore.WHITE}[ 0.0] StochRSI N/A{Style.RESET_ALL}"
        reasons.append(reason_str)

        # --- Final Score Calculation ---
        # Round the raw score to the nearest integer
        final_score = int(round(score))
        logger.debug(f"Signal score calculation complete. Raw Score: {score:.2f}, Final Integer Score: {final_score}")

        return final_score, reasons

    @retry_api_call(max_retries=2, initial_delay=2)  # Retry order placement a couple of times
    def place_entry_order(
        self,
        side: str,
        order_size_base: float,
        confidence_level: int,
        order_type: str,
        current_price: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict[str, Any] | None:
        """Places an entry order (market or limit) on the exchange.
        Includes Bybit V5 specific parameters for SL/TP if provided.

        Args:
            side: 'buy' or 'sell'.
            order_size_base: The desired order size in BASE currency (e.g., DOT amount),
                             already validated and precision-adjusted.
            confidence_level: The signal score associated with this entry.
            order_type: 'market' or 'limit'.
            current_price: The current market price (used for limit offset & logging).
            stop_loss_price: The calculated stop loss price (float, precise) to send with the order.
            take_profit_price: The calculated take profit price (float, precise) to send with the order.

        Returns:
            The CCXT order dictionary if the order was placed successfully (or simulated),
            otherwise None. Includes a 'bot_custom_info' key with confidence level.
        """
        if self.market_info is None:
            logger.error("Market info not available. Cannot place entry order.")
            return None
        if order_size_base <= 0:
            logger.error(f"Order size ({order_size_base}) is zero or negative. Cannot place entry order.")
            return None
        if current_price <= 1e-12:
            logger.error("Current price is zero or negative. Cannot place entry order.")
            return None

        base_currency = self.market_info["base"]
        quote_currency = self.market_info["quote"]
        params = {}  # Dictionary for extra CCXT parameters (like SL/TP, category)
        limit_price: float | None = None  # For limit orders

        try:
            # --- Amount is already precise and limit-checked from calculate_order_size ---
            amount_precise = order_size_base
            logger.debug(
                f"Using pre-validated precise entry amount: {amount_precise:.{self.amount_decimals}f} {base_currency}"
            )

            # --- Calculate Precise Limit Price if Applicable ---
            if order_type == "limit":
                offset = (
                    self.limit_order_entry_offset_pct_buy if side == "buy" else self.limit_order_entry_offset_pct_sell
                )
                price_factor = (1 - offset) if side == "buy" else (1 + offset)
                limit_price_raw = current_price * price_factor
                if limit_price_raw <= 1e-12:
                    logger.error(
                        f"Calculated limit price ({limit_price_raw}) is zero or negative. Cannot place limit order."
                    )
                    return None
                limit_price = float(self.exchange.price_to_precision(self.symbol, limit_price_raw))
                logger.debug(f"Calculated precise limit price: {limit_price:.{self.price_decimals}f} {quote_currency}")
                # Add a small safety check for limit price vs current price
                if (side == "buy" and limit_price >= current_price) or (
                    side == "sell" and limit_price <= current_price
                ):
                    logger.warning(
                        f"{Fore.YELLOW}Calculated limit price {format_price(limit_price)} is not favorable compared to current price {format_price(current_price)} for {side.upper()} order. Proceeding, but check offset.{Style.RESET_ALL}"
                    )

            # --- Add Bybit V5 SL/TP & Trigger Parameters ---
            # Prices should already be precise floats from _calculate_sl_tp_prices
            # Bybit V5 API often expects SL/TP prices as strings in the 'params' dictionary
            if stop_loss_price is not None:
                # Format SL price to string with required precision
                sl_str = f"{stop_loss_price:.{self.price_decimals}f}"
                params["stopLoss"] = sl_str
                if self.sl_trigger_by:
                    params["slTriggerBy"] = self.sl_trigger_by
                logger.debug(f"Adding SL param: stopLoss={params['stopLoss']}, slTriggerBy={params.get('slTriggerBy')}")
            if take_profit_price is not None:
                # Format TP price to string
                tp_str = f"{take_profit_price:.{self.price_decimals}f}"
                params["takeProfit"] = tp_str
                if self.tp_trigger_by:
                    params["tpTriggerBy"] = self.tp_trigger_by
                logger.debug(
                    f"Adding TP param: takeProfit={params['takeProfit']}, tpTriggerBy={params.get('tpTriggerBy')}"
                )

            # Add Bybit V5 category parameter for derivatives if needed
            if self.exchange_id.lower() == "bybit":
                market_type = self.exchange.market(self.symbol).get("type", "swap")  # Assume swap if not specified
                is_linear = self.exchange.market(self.symbol).get("linear", True)  # Assume linear if not specified
                if market_type in ["swap", "future"]:
                    params["category"] = "linear" if is_linear else "inverse"
                    logger.debug(f"Adding Bybit category param: {params['category']}")
                # elif market_type == 'spot': params['category'] = 'spot' # If needed for spot SL/TP?

            # Add other potential params if required by exchange/ccxt version
            # params['timeInForce'] = 'GTC' # Good Till Cancelled (often default)
            # params['postOnly'] = False # Ensure it's not accidentally post-only if limit

        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(
                f"{Fore.RED}Error preparing entry order values/parameters: {e}{Style.RESET_ALL}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error preparing entry order: {e}{Style.RESET_ALL}", exc_info=True)
            return None

        # --- Log Order Details Before Placing ---
        log_color = Fore.GREEN if side == "buy" else Fore.RED
        action_desc = f"{order_type.upper()} {side.upper()} ENTRY"
        sl_info = f"SL={params.get('stopLoss', 'N/A')}" + (
            f" ({params['slTriggerBy']})" if "slTriggerBy" in params else ""
        )
        tp_info = f"TP={params.get('takeProfit', 'N/A')}" + (
            f" ({params['tpTriggerBy']})" if "tpTriggerBy" in params else ""
        )
        limit_price_info = (
            f"Limit={limit_price:.{self.price_decimals}f}"
            if limit_price
            else f"Market (~{current_price:.{self.price_decimals}f})"
        )
        estimated_value_quote = amount_precise * (limit_price or current_price)

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_id = f"sim_entry_{int(time.time() * 1000)}_{side[:1]}"
            # Simulate fill price: use limit price if set, otherwise current price
            sim_entry_price = limit_price if order_type == "limit" else current_price
            # Simulate status: 'open' for limit, 'closed' (filled) for market
            sim_status = "open" if order_type == "limit" else "closed"
            sim_filled = amount_precise if sim_status == "closed" else 0.0
            sim_remaining = 0.0 if sim_status == "closed" else amount_precise
            sim_cost = sim_filled * sim_entry_price if sim_status == "closed" else 0.0
            sim_timestamp_ms = int(time.time() * 1000)
            sim_datetime = pd.to_datetime(sim_timestamp_ms, unit="ms", utc=True).isoformat()

            simulated_order = {
                "id": sim_id,
                "clientOrderId": sim_id,
                "timestamp": sim_timestamp_ms,
                "datetime": sim_datetime,
                "lastTradeTimestamp": sim_timestamp_ms if sim_status == "closed" else None,
                "symbol": self.symbol,
                "type": order_type,
                "side": side,
                "price": limit_price,  # Price of the limit order itself
                "amount": amount_precise,  # Requested amount
                "cost": sim_cost,  # Filled cost
                "average": sim_entry_price if sim_status == "closed" else None,  # Fill price
                "filled": sim_filled,
                "remaining": sim_remaining,
                "status": sim_status,
                "fee": None,
                "fees": [],  # Simulate no fees for simplicity
                "stopLossPrice": stop_loss_price,  # Reflects what *would* have been sent
                "takeProfitPrice": take_profit_price,
                "trades": [],
                "info": {  # Mimic CCXT structure and add simulation details
                    "simulated": True,
                    "orderId": sim_id,
                    "confidence": confidence_level,
                    "initial_base_size": order_size_base,  # Store original calculation base if needed
                    "stopLoss": params.get("stopLoss"),  # Store string param if sent
                    "takeProfit": params.get("takeProfit"),
                    "slTriggerBy": params.get("slTriggerBy"),
                    "tpTriggerBy": params.get("tpTriggerBy"),
                    "category": params.get("category"),
                    "reduceOnly": params.get("reduceOnly", False),
                },
            }
            logger.info(
                f"{log_color}[SIMULATION] Placing {action_desc}: "
                f"ID: {sim_id}, Size: {amount_precise:.{self.amount_decimals}f} {base_currency}, "
                f"Price: {limit_price_info}, "
                f"Est. Value: {estimated_value_quote:.{self.price_decimals}f} {quote_currency}, Confidence: {confidence_level}, {sl_info}, {tp_info}{Style.RESET_ALL}"
            )
            # Add custom info directly for state management consistency
            simulated_order["bot_custom_info"] = {"confidence": confidence_level, "initial_base_size": order_size_base}
            return simulated_order

        # --- Live Trading Mode ---
        else:
            logger.info(f"{log_color}Attempting to place LIVE {action_desc} order...")
            logger.info(f" -> Size: {amount_precise:.{self.amount_decimals}f} {base_currency}")
            logger.info(f" -> Price: {limit_price_info}")
            logger.info(f" -> Value: ~{estimated_value_quote:.{self.price_decimals}f} {quote_currency}")
            logger.info(f" -> Params: {params}")  # Includes SL/TP, category etc.

            order: dict[str, Any] | None = None
            try:
                if order_type == "market":
                    # Note: Bybit V5 might not accept price for market orders, rely on params
                    # Price argument is typically ignored for market orders by CCXT create_market_order
                    order = self.exchange.create_market_order(self.symbol, side, amount_precise, params=params)
                elif order_type == "limit":
                    if limit_price is None:  # Should not happen due to earlier checks
                        logger.error(f"{Fore.RED}Limit price is None. Cannot place live limit order.{Style.RESET_ALL}")
                        return None
                    order = self.exchange.create_limit_order(
                        self.symbol, side, amount_precise, limit_price, params=params
                    )
                else:
                    # Should not happen due to config validation
                    logger.error(f"Unsupported order type '{order_type}' for live trading.")
                    return None

                # Process the order response
                if order:
                    oid = order.get("id", "N/A")
                    otype = order.get("type", "N/A")
                    oside = (order.get("side") or "N/A").upper()
                    oamt = order.get("amount", 0.0)
                    ofilled = order.get("filled", 0.0)
                    oprice = order.get("price")  # Original limit price
                    oavg = order.get("average")  # Average fill price (if filled)
                    ocost = order.get("cost", 0.0)
                    ostatus = order.get("status", STATUS_UNKNOWN)

                    # Try to get SL/TP info back from the order response's 'info' field for confirmation
                    info_sl = order.get("info", {}).get("stopLoss", "N/A")
                    info_tp = order.get("info", {}).get("takeProfit", "N/A")
                    info_sl_trig = order.get("info", {}).get("slTriggerBy", "N/A")
                    info_tp_trig = order.get("info", {}).get("tpTriggerBy", "N/A")

                    oamt_f = f"{oamt:.{self.amount_decimals}f}" if oamt is not None else "N/A"
                    ofilled_f = f"{ofilled:.{self.amount_decimals}f}" if ofilled is not None else "N/A"
                    oprice_f = f"{oprice:.{self.price_decimals}f}" if oprice is not None else "N/A"
                    oavg_f = f"{oavg:.{self.price_decimals}f}" if oavg is not None else "N/A"
                    ocost_f = f"{ocost:.{self.price_decimals}f}" if ocost is not None else "N/A"

                    log_price_details = f"Limit: {oprice_f}" if oprice else "Market"
                    if oavg:
                        log_price_details += f", AvgFill: {oavg_f}"

                    logger.info(
                        f"{log_color}---> LIVE {action_desc} Order Placed: "
                        f"ID: {oid}, Type: {otype}, Side: {oside}, "
                        f"Amount: {oamt_f}, Filled: {ofilled_f}, "
                        f"{log_price_details}, Cost: {ocost_f} {quote_currency}, "
                        f"Status: {ostatus}, "
                        f"SL Sent/Confirmed: {params.get('stopLoss', 'N/A')}/{info_sl} ({info_sl_trig}), "
                        f"TP Sent/Confirmed: {params.get('takeProfit', 'N/A')}/{info_tp} ({info_tp_trig}), "
                        f"Confidence: {confidence_level}{Style.RESET_ALL}"
                    )
                    # Store bot-specific context with the order dict before returning
                    order["bot_custom_info"] = {"confidence": confidence_level, "initial_base_size": order_size_base}
                    return order
                else:
                    # API call succeeded but returned None/empty response?
                    logger.error(
                        f"{Fore.RED}LIVE {action_desc} order placement potentially failed: API call succeeded but returned None or empty response. Check exchange status/order history manually.{Style.RESET_ALL}"
                    )
                    logger.error(f" -> Params Sent: {params}")
                    return None

            # Handle specific, potentially non-retryable errors based on decorator logic
            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                # These are often logged as ERROR by the retry decorator if non-retryable
                logger.error(
                    f"{Fore.RED}LIVE {action_desc} Order Failed ({type(e).__name__}): {e}. Check balance or order parameters.{Style.RESET_ALL}"
                )
                logger.error(f" -> Params Sent: {params}")
                # Re-fetch balance after insufficient funds error for logging context
                if isinstance(e, ccxt.InsufficientFunds):
                    self.fetch_balance(quote_currency)
                return None
            except ccxt.ExchangeError as e:
                # Catch other exchange errors not handled by specific cases or decorator
                logger.error(f"{Fore.RED}LIVE {action_desc} Order Failed (ExchangeError): {e}{Style.RESET_ALL}")
                logger.error(f" -> Params Sent: {params}")
                return None
            except Exception as e:
                # Catch any unexpected Python errors
                logger.error(
                    f"{Fore.RED}LIVE {action_desc} Order Failed (Unexpected Python Error): {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                logger.error(f" -> Params Sent: {params}")
                return None

    @retry_api_call(max_retries=1)  # Only retry cancellation once if it fails initially
    def cancel_order_by_id(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancels a single open order by its ID.

        Args:
            order_id: The exchange's ID for the order to cancel.
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            True if the order was successfully cancelled or already not found/closed/filled,
            False otherwise (e.g., network error after retry, unexpected exchange error).
        """
        if not order_id:
            logger.warning("cancel_order_by_id called with empty order_id.")
            return False

        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}Attempting to cancel order {order_id} ({target_symbol})...{Style.RESET_ALL}")

        # --- Simulation Mode ---
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Simulating cancellation of order {order_id}.")
            found_and_updated = False
            needs_save = False
            for i, pos in enumerate(self.open_positions):
                # Only cancel orders that are actually pending entry
                if pos["id"] == order_id and pos["status"] == STATUS_PENDING_ENTRY:
                    self.open_positions[i]["status"] = STATUS_CANCELED
                    self.open_positions[i]["last_update_time"] = time.time()
                    logger.info(f"[SIMULATION] Marked simulated order {order_id} as canceled.")
                    found_and_updated = True
                    needs_save = True
                    break
            if not found_and_updated:
                logger.warning(
                    f"[SIMULATION] Simulated order {order_id} not found or not in cancellable state ({STATUS_PENDING_ENTRY}). Assuming cancellation is not needed or already done."
                )
            # In simulation, return True signifies the intent was processed or unnecessary
            if needs_save:
                self._save_state()
            return True

        # --- Live Mode ---
        else:
            try:
                # Add params if needed (e.g., Bybit V5 category)
                params = {}
                if self.exchange_id.lower() == "bybit":
                    market_type = self.exchange.market(target_symbol).get("type", "swap")
                    is_linear = self.exchange.market(target_symbol).get("linear", True)
                    if market_type in ["swap", "future"]:
                        params["category"] = "linear" if is_linear else "inverse"
                    elif market_type == "spot":
                        params["category"] = "spot"

                # Use unified cancel_order method
                response = self.exchange.cancel_order(order_id, target_symbol, params=params)
                # Response format varies, but success usually means no exception
                logger.info(
                    f"{Fore.YELLOW}---> Successfully initiated cancellation for order {order_id}. Response snippet: {str(response)[:100]}...{Style.RESET_ALL}"
                )
                return True
            except ccxt.OrderNotFound:
                # If the order is already gone (filled, cancelled, expired), consider it a success.
                logger.warning(
                    f"{Fore.YELLOW}Order {order_id} not found during cancellation attempt (already closed/cancelled?). Treating as success.{Style.RESET_ALL}"
                )
                return True
            except ccxt.NetworkError as e:
                # Let the retry decorator handle transient network issues
                logger.warning(f"Network error cancelling order {order_id}: {e}. Retrying if possible.")
                raise e  # Re-raise to trigger retry
            except ccxt.ExchangeError as e:
                # Log specific exchange errors during cancellation more prominently
                err_str = str(e).lower()
                # Check for common phrases indicating the order is already in a final state
                if any(
                    phrase in err_str
                    for phrase in [
                        "order has been filled",
                        "order is finished",
                        "already closed",
                        "already cancelled",
                        "order status error",
                        "cannot be cancelled",
                    ]
                ):
                    logger.warning(
                        f"{Fore.YELLOW}Cannot cancel order {order_id}: Already filled, finished, or in non-cancellable state. Error: {e}{Style.RESET_ALL}"
                    )
                    return True  # Treat as 'success' in the sense that it doesn't need cancelling / action completed
                else:
                    # Log other exchange errors as failures
                    logger.error(f"{Fore.RED}Exchange error cancelling order {order_id}: {e}{Style.RESET_ALL}")
                    logger.error(f" -> Params Sent: {params}")
                    return False  # Cancellation failed due to exchange state/rules
            except Exception as e:
                # Catch unexpected Python errors
                logger.error(
                    f"{Fore.RED}Unexpected error cancelling order {order_id}: {e}{Style.RESET_ALL}", exc_info=True
                )
                return False

    @retry_api_call()  # Retry fetching open orders or cancelAll
    def cancel_all_symbol_orders(self, symbol: str | None = None) -> int:
        """Attempts to cancel all open orders for the specified symbol.
        Prefers `fetchOpenOrders` and individual cancellation for better feedback,
        falls back to `cancelAllOrders` if available and necessary.

        Args:
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            The number of orders successfully cancelled (or -1 if `cancelAllOrders` was used
            and count is unknown). Returns 0 if no orders were found or cancellation failed.
        """
        target_symbol = symbol or self.symbol
        logger.info(
            f"{Fore.YELLOW}Checking for and attempting to cancel all OPEN orders for {target_symbol}...{Style.RESET_ALL}"
        )
        cancelled_count = 0

        # --- Simulation Mode ---
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Simulating cancellation of all PENDING orders for {target_symbol}.")
            sim_cancelled = 0
            needs_save = False
            for i, pos in enumerate(self.open_positions):
                if pos.get("symbol") == target_symbol and pos.get("status") == STATUS_PENDING_ENTRY:
                    self.open_positions[i]["status"] = STATUS_CANCELED
                    self.open_positions[i]["last_update_time"] = time.time()
                    sim_cancelled += 1
                    needs_save = True
            if sim_cancelled > 0:
                logger.info(f"[SIMULATION] Marked {sim_cancelled} pending simulated orders as canceled.")
                if needs_save:
                    self._save_state()
            else:
                logger.info(f"[SIMULATION] No pending simulated orders found for {target_symbol} to cancel.")
            return sim_cancelled

        # --- Live Mode ---
        else:
            try:
                # Prefer fetching open orders and cancelling individually for better feedback
                if self.exchange.has["fetchOpenOrders"]:
                    logger.debug(f"Fetching open orders for {target_symbol} to cancel individually...")
                    params = {}
                    if self.exchange_id.lower() == "bybit":
                        market_type = self.exchange.market(target_symbol).get("type", "swap")
                        is_linear = self.exchange.market(target_symbol).get("linear", True)
                        if market_type in ["swap", "future"]:
                            params["category"] = "linear" if is_linear else "inverse"
                        elif market_type == "spot":
                            params["category"] = "spot"

                    # Fetch only open orders (not closed/filled/etc.)
                    open_orders = self.exchange.fetch_open_orders(target_symbol, params=params)

                    if not open_orders:
                        logger.info(f"No currently open orders found for {target_symbol} via fetchOpenOrders.")
                        return 0

                    logger.warning(
                        f"{Fore.YELLOW}Found {len(open_orders)} open order(s) for {target_symbol}. Attempting individual cancellation...{Style.RESET_ALL}"
                    )
                    for order in open_orders:
                        order_id = order.get("id")
                        if not order_id:
                            logger.warning(f"Found open order without ID: {str(order)[:100]}... Skipping cancellation.")
                            continue

                        # Use the individual cancel function (which handles simulation/live & retries)
                        if self.cancel_order_by_id(order_id, target_symbol):
                            cancelled_count += 1
                            # Add small delay to potentially avoid rate limits if cancelling many orders quickly
                            time.sleep(max(0.2, self.exchange.rateLimit / 1000))  # Use min 0.2s delay
                        else:
                            logger.error(
                                f"Failed to cancel order {order_id} during bulk cancellation attempt. Continuing..."
                            )
                    logger.info(
                        f"Individual cancellation attempt finished. Successfully cancelled: {cancelled_count}/{len(open_orders)}."
                    )

                # Fallback to cancelAllOrders if fetchOpenOrders is not supported or preferred
                elif self.exchange.has["cancelAllOrders"]:
                    logger.warning(
                        f"{Fore.YELLOW}Exchange lacks 'fetchOpenOrders' or it failed, attempting unified 'cancelAllOrders' for {target_symbol}...{Style.RESET_ALL}"
                    )
                    params = {}
                    if self.exchange_id.lower() == "bybit":
                        market_type = self.exchange.market(target_symbol).get("type", "swap")
                        is_linear = self.exchange.market(target_symbol).get("linear", True)
                        if market_type in ["swap", "future"]:
                            params["category"] = "linear" if is_linear else "inverse"
                        elif market_type == "spot":
                            params["category"] = "spot"
                        # Bybit might require 'orderFilter=Order' to only cancel regular orders? Check docs.
                        # params['orderFilter'] = 'Order'

                    response = self.exchange.cancel_all_orders(target_symbol, params=params)
                    logger.info(f"'cancelAllOrders' for {target_symbol} response snippet: {str(response)[:100]}...")
                    # We don't know the exact count from this response typically
                    cancelled_count = -1  # Indicate unknown count, but action was attempted
                else:
                    # If neither method is supported, we cannot proceed
                    logger.error(
                        f"{Fore.RED}Exchange {self.exchange_id} does not support 'fetchOpenOrders' or 'cancelAllOrders' for {target_symbol}. Cannot cancel all orders automatically.{Style.RESET_ALL}"
                    )
                    return 0  # Return 0 as no action could be taken

                final_msg = f"Bulk order cancellation process finished for {target_symbol}. "
                if cancelled_count >= 0:
                    final_msg += f"Attempted/Cancelled: {cancelled_count} order(s)."
                else:
                    final_msg += "'cancelAllOrders' command sent (exact count unknown)."
                logger.info(final_msg)
                return cancelled_count

            except Exception as e:
                # Catch any other errors during the fetch or cancel loop
                logger.error(
                    f"{Fore.RED}Error during bulk order cancellation process for {target_symbol}: {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                return cancelled_count  # Return count successfully cancelled before error

    # --- Position Management & State Update Logic ---

    def _check_pending_entries(self, indicators: dict) -> None:
        """Checks the status of any pending limit entry orders (status='pending_entry').
        If an order is filled, updates its status to 'active', records actual entry
        price/time, and recalculates/stores internal SL/TP based on the fill price.
        Removes failed (canceled/rejected) or vanished pending orders from the state.

        Args:
            indicators: Dictionary containing current indicator values (e.g., ATR for SL/TP recalc).
        """
        # Operate on a copy to avoid modification issues during iteration
        pending_positions = [pos for pos in self.open_positions if pos.get("status") == STATUS_PENDING_ENTRY]
        if not pending_positions:
            return  # No pending orders to check

        logger.debug(f"Checking status of {len(pending_positions)} pending entry order(s)...")
        current_price_for_check: float | None = None  # Fetch only if needed
        needs_state_save = False
        positions_to_remove_ids = set()
        positions_to_update_data = {}  # Dict[order_id, updated_position_dict]

        for position in pending_positions:
            entry_order_id = position.get("id")
            pos_symbol = position.get("symbol", self.symbol)  # Use position's symbol

            if not entry_order_id:
                logger.error(f"Found pending position with no ID: {str(position)[:100]}... Skipping.")
                continue

            order_info = self.fetch_order_status(entry_order_id, symbol=pos_symbol)

            # Case 1: Order Not Found or Finished Externally
            if order_info is None:
                logger.warning(
                    f"{Fore.YELLOW}Pending entry order {entry_order_id} ({pos_symbol}) status unknown/not found. Assuming closed/cancelled externally. Removing from pending state.{Style.RESET_ALL}"
                )
                positions_to_remove_ids.add(entry_order_id)
                needs_state_save = True
                continue

            order_status = order_info.get("status")
            filled_amount = float(order_info.get("filled", 0.0))
            entry_price = order_info.get("average")  # Use average fill price if available

            # Case 2: Order Fully Filled ('closed' status, filled > 0)
            if order_status == "closed" and filled_amount > 1e-12:
                if entry_price is None or float(entry_price) <= 1e-12:
                    # If closed but no valid fill price, try getting last trade price or current price? Risky.
                    logger.error(
                        f"{Fore.RED}Entry order {entry_order_id} ({pos_symbol}) is 'closed' but has invalid/missing avg fill price ({entry_price}). Cannot activate reliably. Removing.{Style.RESET_ALL}"
                    )
                    positions_to_remove_ids.add(entry_order_id)
                    needs_state_save = True
                    continue

                entry_price = float(entry_price)  # Ensure float

                # Compare filled amount to originally requested amount
                orig_size = position.get("original_size")
                if orig_size is not None and abs(filled_amount - orig_size) > max(
                    orig_size * 0.01, 1e-9
                ):  # Allow 1% tolerance
                    logger.warning(
                        f"{Fore.YELLOW}Entry order {entry_order_id} filled amount ({filled_amount:.{self.amount_decimals}f}) differs from requested ({orig_size:.{self.amount_decimals}f}). Using actual filled amount.{Style.RESET_ALL}"
                    )

                logger.info(
                    f"{Fore.GREEN}---> Pending entry order {entry_order_id} ({pos_symbol}) FILLED! Amount: {filled_amount:.{self.amount_decimals}f}, Avg Price: {entry_price:.{self.price_decimals}f}{Style.RESET_ALL}"
                )

                # --- Update Position State ---
                updated_pos = position.copy()
                updated_pos["size"] = filled_amount  # Update size to actual filled amount
                updated_pos["entry_price"] = entry_price
                updated_pos["status"] = STATUS_ACTIVE
                # Use fill timestamp from order if available, else use current time
                # CCXT 'lastTradeTimestamp' might be more accurate if available
                fill_time_ms = order_info.get("lastTradeTimestamp") or order_info.get("timestamp")
                updated_pos["entry_time"] = fill_time_ms / 1000 if fill_time_ms else time.time()
                updated_pos["last_update_time"] = time.time()

                # --- Recalculate and Store Internal SL/TP based on Actual Fill Price ---
                # Fetch current price if needed for the recalculation
                if current_price_for_check is None:
                    current_price_for_check = self.fetch_market_price()

                if current_price_for_check:
                    atr_value = indicators.get("atr")  # Get current ATR from passed dict
                    sl_price, tp_price = self._calculate_sl_tp_prices(
                        entry_price=updated_pos["entry_price"],  # Use actual fill price
                        side=updated_pos["side"],
                        current_price=current_price_for_check,  # Use current price for sanity checks
                        atr=atr_value,
                    )
                    # Store these potentially adjusted SL/TP prices in the bot's internal state
                    # These are the prices the bot will use for TSL or other internal logic.
                    # We assume the exchange's parameter-based SL/TP handles itself based on the initial request parameters.
                    updated_pos["stop_loss_price"] = sl_price
                    updated_pos["take_profit_price"] = tp_price
                    logger.info(
                        f"Stored internal SL={format_price(sl_price)}, TP={format_price(tp_price)} for activated pos {entry_order_id} based on actual entry {entry_price:.{self.price_decimals}f}."
                    )
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Could not fetch current price. SL/TP recalculation skipped after entry fill for {entry_order_id}. Internal SL/TP might be based on original estimate.{Style.RESET_ALL}"
                    )
                    # Keep SL/TP prices that were originally stored when position was created (based on estimate)

                positions_to_update_data[entry_order_id] = updated_pos
                needs_state_save = True

            # Case 3: Order Failed (canceled, rejected, expired)
            elif order_status in ["canceled", "rejected", "expired"]:
                reason = order_info.get("info", {}).get(
                    "rejectReason", order_info.get("info", {}).get("cancelType", "No reason provided")
                )  # Try Bybit fields
                logger.warning(
                    f"{Fore.YELLOW}Pending entry order {entry_order_id} ({pos_symbol}) failed (Status: {order_status}, Reason: {reason}). Removing from state.{Style.RESET_ALL}"
                )
                positions_to_remove_ids.add(entry_order_id)
                needs_state_save = True

            # Case 4: Order Still Open or Partially Filled (remains pending)
            elif order_status == "open":
                filled_f = f"{filled_amount:.{self.amount_decimals}f}"
                logger.debug(
                    f"Pending entry order {entry_order_id} ({pos_symbol}) is still 'open'. Filled: {filled_f}. Waiting..."
                )
                # Optionally update last_update_time for the pending order
                # position['last_update_time'] = time.time() # Needs to modify original list or use update dict
                # positions_to_update_data[entry_order_id] = {'last_update_time': time.time()} # Safer?
            elif order_status == "closed" and filled_amount <= 1e-12:
                # Can mean cancelled before any fill occurred, or rejected
                logger.warning(
                    f"{Fore.YELLOW}Pending entry order {entry_order_id} ({pos_symbol}) has status 'closed' but zero filled amount. Assuming cancelled/rejected pre-fill. Removing.{Style.RESET_ALL}"
                )
                positions_to_remove_ids.add(entry_order_id)
                needs_state_save = True
            else:
                # Unexpected status for a pending order
                logger.warning(
                    f"Pending entry order {entry_order_id} ({pos_symbol}) has unexpected status: {order_status}. Filled: {filled_amount}. Leaving as pending for now."
                )

        # --- Apply State Updates Atomically ---
        if needs_state_save:
            len(self.open_positions)
            new_positions_list = []
            removed_count = 0
            updated_count = 0

            for pos in self.open_positions:
                pos_id = pos.get("id")
                if pos_id in positions_to_remove_ids:
                    removed_count += 1
                    continue  # Skip removed positions

                if pos_id in positions_to_update_data:
                    # Replace with the fully updated dict (contains fill info, new status, SL/TP)
                    new_positions_list.append(positions_to_update_data[pos_id])
                    updated_count += 1
                else:
                    # Keep unchanged or only timestamp-updated positions
                    # if pos_id in positions_to_update_data: # Apply timestamp update if any
                    #      pos.update(positions_to_update_data[pos_id])
                    new_positions_list.append(pos)

            self.open_positions = new_positions_list
            logger.debug(
                f"Pending checks complete. Filled/Activated: {updated_count}, Removed/Failed: {removed_count}. Current total positions: {len(self.open_positions)}"
            )
            self._save_state()  # Save the changes

    def _manage_active_positions(self, current_price: float, indicators: dict) -> None:
        """Manages currently active positions (status='active').

        - Checks if the position was closed externally (e.g., SL/TP hit via parameters, manual closure).
          Uses `fetch_order_status` on the entry order ID as primary check. Might need refinement
          if exchange handles SL/TP via separate orders (consider `fetch_positions` if available/needed).
        - Implements Time-Based Exit logic by placing a market close order.
        - Implements **EXPERIMENTAL** Trailing Stop Loss (TSL) logic using `edit_order`.
          **VERIFY EXCHANGE/CCXT SUPPORT FOR `edit_order` with SL/TP modification.**
        - Updates position state (e.g., TSL price) or removes closed positions, calculating PnL.

        Args:
            current_price: The current market price.
            indicators: Dictionary containing current indicator values (e.g., ATR).
        """
        # Operate on a copy to avoid modification issues during iteration
        active_positions = [pos for pos in self.open_positions if pos.get("status") == STATUS_ACTIVE]
        if not active_positions:
            return

        logger.debug(
            f"Managing {len(active_positions)} active position(s) against current price: {current_price:.{self.price_decimals}f}..."
        )
        needs_state_save = False
        positions_to_remove_ids = set()
        positions_to_update_state = {}  # Store state updates {pos_id: {key: value}}

        for position in active_positions:
            pos_id = position.get("id")  # ID of the original entry order
            if not pos_id:
                logger.error(f"Active position missing ID: {position}")
                continue

            symbol = position.get("symbol", self.symbol)
            side = position.get("side")
            entry_price = position.get("entry_price")  # Should be set when status became ACTIVE
            position_size = position.get("size")  # Should be actual filled size
            entry_time = position.get("entry_time", time.time())  # Should be fill time
            position.get("stop_loss_price")  # Bot's internal SL
            position.get("take_profit_price")  # Bot's internal TP
            position.get("trailing_stop_price")  # Bot's internal TSL

            # Basic validation of essential data for an active position
            if not all([symbol, side, entry_price, position_size, entry_time]):
                logger.error(
                    f"Active position {pos_id} ({symbol}) has incomplete essential data: entry={entry_price}, size={position_size}, time={entry_time}. Skipping management."
                )
                continue
            if position_size <= 1e-12:
                logger.warning(
                    f"{Fore.YELLOW}Active position {pos_id} ({symbol}) has size ~0 ({position_size}). Assuming closed. Removing.{Style.RESET_ALL}"
                )
                positions_to_remove_ids.add(pos_id)
                needs_state_save = True
                continue

            exit_reason: str | None = None
            exit_price: float | None = None  # Price at which the position closed

            # --- 1. Check Status of the Original Entry Order (Primary Check for External Closure) ---
            # With parameter-based SL/TP, the exchange *might* close the original order, or create new ones.
            # If this check is unreliable, consider using `fetch_positions` if supported and reliable.
            order_info = self.fetch_order_status(pos_id, symbol=symbol)

            if order_info is None:
                # Order not found - Assume closed externally if the position was marked active in our state.
                logger.warning(
                    f"{Fore.YELLOW}Active position's main order {pos_id} ({symbol}) not found. Assuming closed/cancelled externally. Removing position.{Style.RESET_ALL}"
                )
                exit_reason = f"Main Order Vanished ({pos_id})"
                exit_price = current_price  # Best guess is current price

            elif order_info.get("status") == "closed":
                # Check if it was filled previously (entry) or if it represents a closure event
                # If filled amount roughly matches position size, it's likely the entry fill confirmation.
                # If filled amount is zero or much smaller, it might indicate a close event (e.g., SL/TP hit).
                # This logic is imperfect and depends on how the exchange reports SL/TP fills via the original order ID.
                order_filled = float(order_info.get("filled", 0.0))
                # If the order is closed and the filled amount matches our known position size,
                # it's likely just confirming the entry fill we already processed.
                # If filled is 0 or different, assume it signals a closure event (SL/TP/Manual).
                if abs(order_filled - position_size) > max(position_size * 0.01, 1e-9):  # Allow 1% tolerance
                    # Infer closure reason
                    logger.info(
                        f"Order {pos_id} ({symbol}) status is 'closed', filled amount ({order_filled}) differs from position size ({position_size}). Inferring external closure."
                    )
                    exit_price_raw = (
                        order_info.get("average") or order_info.get("price") or current_price
                    )  # Use average if available (unlikely for closure)
                    exit_price = float(exit_price_raw) if exit_price_raw is not None else current_price

                    # Try inferring reason from Bybit V5 info fields or price comparison
                    exit_reason = self._infer_exit_reason(order_info, position, exit_price, current_price)
                    log_color = (
                        Fore.RED if "SL" in exit_reason else (Fore.GREEN if "TP" in exit_reason else Fore.YELLOW)
                    )
                    logger.info(f"{log_color}{exit_reason} at ~{exit_price:.{self.price_decimals}f}{Style.RESET_ALL}")
                else:
                    # Order is closed and filled amount matches -> Likely just confirming entry, position is still active in reality.
                    logger.debug(
                        f"Order {pos_id} status 'closed' matches filled size. Position likely still active. No exit action taken based on this."
                    )

            elif order_info.get("status") in ["canceled", "rejected", "expired"]:
                # Should not happen for an 'active' position's main order ID. Indicates state mismatch or external action.
                logger.error(
                    f"{Fore.RED}State inconsistency! Active position {pos_id} ({symbol}) linked to order with status '{order_info.get('status')}'. Assuming closed. Removing position.{Style.RESET_ALL}"
                )
                exit_reason = f"Order Inconsistency ({order_info.get('status')})"
                exit_price = current_price

            # --- If order seems active or status doesn't indicate closure, check bot-managed exits ---
            # Note: if `fetch_order_status` is unreliable for SL/TP hits, `fetch_positions` might be needed here.
            if not exit_reason:
                # --- 2. Time-Based Exit Check ---
                if self.time_based_exit_minutes is not None and self.time_based_exit_minutes > 0:
                    time_elapsed_seconds = time.time() - entry_time
                    time_elapsed_minutes = time_elapsed_seconds / 60
                    if time_elapsed_minutes >= self.time_based_exit_minutes:
                        logger.info(
                            f"{Fore.YELLOW}Time limit ({self.time_based_exit_minutes} min) reached for active position {pos_id} (Age: {time_elapsed_minutes:.1f}m). Initiating market close.{Style.RESET_ALL}"
                        )
                        market_close_order = self._place_market_close_order(position, current_price)
                        if market_close_order:
                            exit_reason = f"Time Limit Reached ({self.time_based_exit_minutes}min)"
                            # Use fill price from close order if available, else current price
                            exit_price = market_close_order.get("average") or current_price
                            logger.info(
                                f"Market close order for time exit placed successfully for {pos_id}. Exit Price ~{format_price(exit_price)}"
                            )
                        else:
                            logger.critical(
                                f"{Fore.RED}{Style.BRIGHT}CRITICAL FAILURE: Failed to place market close order for time-based exit of position {pos_id}. POSITION REMAINS OPEN! Manual intervention required!{Style.RESET_ALL}"
                            )
                            # Do NOT set exit_reason, let it try again or require manual action.
                # --- End Time-Based Exit ---

                # --- 3. Trailing Stop Loss (TSL) Logic [EXPERIMENTAL] ---
                if not exit_reason and self.enable_trailing_stop_loss:
                    if not self.trailing_stop_loss_percentage or not (0 < self.trailing_stop_loss_percentage < 1):
                        logger.error(
                            "TSL enabled but trailing_stop_loss_percentage is invalid. Disabling TSL for this iteration."
                        )
                    else:
                        new_tsl_price = self._update_trailing_stop_price(position, current_price)

                        # If TSL price was updated, attempt EXPERIMENTAL update via edit_order
                        if new_tsl_price is not None:
                            logger.warning(
                                f"{Fore.YELLOW}[EXPERIMENTAL] Attempting TSL update via edit_order for {pos_id} to SL={format_price(new_tsl_price)}. VERIFY EXCHANGE/CCXT SUPPORT!{Style.RESET_ALL}"
                            )

                            edit_success = self._attempt_edit_order_for_tsl(position, new_tsl_price)

                            if edit_success:
                                # Update internal state ONLY if edit seems successful
                                update_payload = {
                                    "trailing_stop_price": new_tsl_price,
                                    "stop_loss_price": new_tsl_price,  # Main internal SL follows TSL
                                    "last_update_time": time.time(),
                                }
                                # Merge with potential previous updates for this pos_id
                                positions_to_update_state[pos_id] = {
                                    **positions_to_update_state.get(pos_id, {}),
                                    **update_payload,
                                }
                                needs_state_save = True
                                logger.info(
                                    f"{Fore.MAGENTA}Internal TSL state for {pos_id} updated to {format_price(new_tsl_price)} after edit attempt.{Style.RESET_ALL}"
                                )
                            else:
                                logger.error(
                                    f"{Fore.RED}TSL update via edit_order for {pos_id} failed or was skipped. TSL state NOT updated.{Style.RESET_ALL}"
                                )
                # --- End TSL Logic ---

            # --- 4. Process Exit and Log PnL ---
            if exit_reason:
                positions_to_remove_ids.add(pos_id)
                needs_state_save = True
                self._log_position_pnl(position, exit_price, exit_reason)

        # --- Apply State Updates and Removals ---
        if needs_state_save:
            len(self.open_positions)
            new_positions_list = []
            removed_count = 0
            updated_count = 0

            for pos in self.open_positions:
                pos_id = pos.get("id")
                if pos_id in positions_to_remove_ids:
                    removed_count += 1
                    continue  # Skip removed positions
                if pos_id in positions_to_update_state:
                    pos.update(positions_to_update_state[pos_id])  # Apply updates
                    updated_count += 1
                new_positions_list.append(pos)  # Keep updated or unchanged positions

            self.open_positions = new_positions_list
            if removed_count > 0 or updated_count > 0:
                logger.debug(
                    f"Active position management complete. Updated: {updated_count}, Removed/Closed: {removed_count}. Current total: {len(self.open_positions)}"
                )
                self._save_state()  # Save changes to state file

    def _infer_exit_reason(self, order_info: dict, position: dict, exit_price: float, current_price: float) -> str:
        """Tries to infer the reason for an external position closure."""
        pos_id = position["id"]
        side = position["side"]
        stored_sl = position.get("stop_loss_price")
        stored_tp = position.get("take_profit_price")
        entry_price = position["entry_price"]
        reason = f"Order Closed Externally ({pos_id})"  # Default

        # Try Bybit V5 specific fields first
        info = order_info.get("info", {})
        bybit_status = info.get("orderStatus", "").lower()
        bybit_stop_order_type = info.get("stopOrderType", "").lower()  # e.g., 'StopLoss', 'TakeProfit'

        if "stop-loss order triggered" in bybit_status or bybit_stop_order_type == "stoploss":
            reason = f"Stop Loss Triggered (Exchange Reported - {pos_id})"
        elif "take-profit order triggered" in bybit_status or bybit_stop_order_type == "takeprofit":
            reason = f"Take Profit Triggered (Exchange Reported - {pos_id})"
        elif "cancel" in bybit_status:
            reason = f"Order Cancelled Externally ({pos_id}, Status: {bybit_status})"
        else:
            # Fallback to price comparison if Bybit info is unclear
            sl_triggered = False
            tp_triggered = False
            # Use a small tolerance (e.g., 0.1% of entry price or a few ticks)
            price_tick = float(self.market_info["precision"]["price"] or 0.0001)
            tolerance = max(entry_price * 0.001, price_tick * 5)

            if stored_sl is not None:
                sl_triggered = (side == "buy" and exit_price <= stored_sl + tolerance) or (
                    side == "sell" and exit_price >= stored_sl - tolerance
                )
            if stored_tp is not None:
                tp_triggered = (side == "buy" and exit_price >= stored_tp - tolerance) or (
                    side == "sell" and exit_price <= stored_tp + tolerance
                )

            if sl_triggered and tp_triggered:  # Ambiguous if prices are very close
                reason = f"SL/TP Hit (Inferred Price Ambiguous - {pos_id})"
            elif sl_triggered:
                reason = f"Stop Loss Hit (Inferred Price - {pos_id})"
            elif tp_triggered:
                reason = f"Take Profit Hit (Inferred Price - {pos_id})"
            else:  # Could be manual close or other reason
                reason = f"Order Closed Externally (Reason Unclear - {pos_id}, Status: {bybit_status})"

        return reason

    def _update_trailing_stop_price(self, position: dict, current_price: float) -> float | None:
        """Calculates if the trailing stop loss needs activation or update.

        Args:
            position: The active position dictionary.
            current_price: The current market price.

        Returns:
            The new trailing stop loss price if an update/activation is needed, else None.
            Returns None if TSL percentage is invalid.
        """
        side = position["side"]
        entry_price = position["entry_price"]
        trailing_sl_price = position.get("trailing_stop_price")
        stored_sl_price = position.get("stop_loss_price")  # Original or last known SL
        tsl_percentage = self.trailing_stop_loss_percentage
        is_tsl_active = trailing_sl_price is not None

        if not tsl_percentage or not (0 < tsl_percentage < 1):
            # Already logged error in caller, just return None here
            return None

        new_tsl_price: float | None = None
        tsl_factor = (1 - tsl_percentage) if side == "buy" else (1 + tsl_percentage)
        potential_tsl_price_raw = current_price * tsl_factor

        # Format potential price for logging/comparison
        try:
            potential_tsl_price = float(self.exchange.price_to_precision(self.symbol, potential_tsl_price_raw))
            if potential_tsl_price <= 0:
                potential_tsl_price = None  # Safety
        except Exception as format_err:
            logger.error(f"Error formatting potential TSL price {potential_tsl_price_raw}: {format_err}")
            potential_tsl_price = None

        if potential_tsl_price is None:
            return None  # Cannot proceed if formatting failed

        # --- TSL Activation Check ---
        if not is_tsl_active:
            # Activation condition: Price must move favorably beyond entry + buffer
            price_tick = float(self.market_info["precision"]["price"] or 0.0001)
            activation_buffer = max(entry_price * 0.001, price_tick * 5)  # e.g., 0.1% or 5 ticks
            activation_price = entry_price + activation_buffer if side == "buy" else entry_price - activation_buffer

            activate_tsl = (side == "buy" and current_price > activation_price) or (
                side == "sell" and current_price < activation_price
            )

            if activate_tsl:
                # Initial TSL: Trail from current price, but ensure it's better than original SL (if set)
                # and better than the activation price itself.
                initial_trail_price = potential_tsl_price
                if stored_sl_price is not None:
                    initial_trail_price = (
                        max(initial_trail_price, stored_sl_price)
                        if side == "buy"
                        else min(initial_trail_price, stored_sl_price)
                    )
                # Ensure TSL doesn't activate worse than activation threshold
                initial_trail_price = (
                    max(initial_trail_price, activation_price)
                    if side == "buy"
                    else min(initial_trail_price, activation_price)
                )

                # Ensure the initial TSL is different from existing SL
                if initial_trail_price != stored_sl_price:
                    new_tsl_price = initial_trail_price
                    logger.info(
                        f"{Fore.MAGENTA}Trailing Stop ACTIVATING for {side.upper()} {position['id']} at {format_price(new_tsl_price)} (Price: {format_price(current_price)}){Style.RESET_ALL}"
                    )
                else:
                    logger.debug(
                        f"TSL activation condition met for {position['id']}, but potential TSL ({format_price(initial_trail_price)}) is same as current SL. No activation needed yet."
                    )

        # --- TSL Update Check ---
        elif is_tsl_active:  # TSL already active
            # Check if price moved further favorably, making potential TSL better than current TSL
            update_needed = (side == "buy" and potential_tsl_price > trailing_sl_price) or (
                side == "sell" and potential_tsl_price < trailing_sl_price
            )

            if update_needed:
                new_tsl_price = potential_tsl_price
                logger.info(
                    f"{Fore.MAGENTA}Trailing Stop UPDATING for {side.upper()} {position['id']} from {format_price(trailing_sl_price)} to {format_price(new_tsl_price)} (Price: {format_price(current_price)}){Style.RESET_ALL}"
                )

        return new_tsl_price

    def _attempt_edit_order_for_tsl(self, position: dict, new_tsl_price: float) -> bool:
        """Attempts to modify the active order's stop loss using `edit_order`.
        **EXPERIMENTAL - Use with caution!**.

        Args:
            position: The position dictionary.
            new_tsl_price: The new stop loss price to set.

        Returns:
            True if the edit order call was attempted and did not raise an immediate
            non-retryable error (like NotSupported). Does NOT guarantee success.
            False otherwise.
        """
        pos_id = position["id"]
        symbol = position["symbol"]
        stored_tp_price = position.get("take_profit_price")  # Get existing TP to potentially resend

        try:
            # --- Prepare Parameters for edit_order ---
            # Parameters for editing SL/TP on Bybit V5 (check CCXT docs for exact names/types)
            edit_params = {
                "stopLoss": f"{new_tsl_price:.{self.price_decimals}f}"  # Send as string for Bybit V5 params
            }
            # To keep existing TP, MUST resend it when modifying SL on Bybit V5
            if stored_tp_price:
                edit_params["takeProfit"] = f"{stored_tp_price:.{self.price_decimals}f}"
                if self.tp_trigger_by:
                    edit_params["tpTriggerBy"] = self.tp_trigger_by
            # Resend SL trigger type
            if self.sl_trigger_by:
                edit_params["slTriggerBy"] = self.sl_trigger_by
            # Bybit V5 category
            if self.exchange_id.lower() == "bybit":
                market_type = self.exchange.market(symbol).get("type", "swap")
                is_linear = self.exchange.market(symbol).get("linear", True)
                if market_type in ["swap", "future"]:
                    edit_params["category"] = "linear" if is_linear else "inverse"
                elif market_type == "spot":
                    edit_params["category"] = "spot"  # If applicable

            # --- Fetch Fresh Order Info for Required edit_order Arguments ---
            # edit_order often needs original type, side, amount, price (for limit)
            # Use retry logic for this fetch
            fresh_order_info = self.fetch_order_status(pos_id, symbol=symbol)
            if not fresh_order_info:
                logger.warning(f"Cannot fetch fresh order info for {pos_id} ({symbol}). Skipping TSL edit attempt.")
                return False
            if fresh_order_info.get("status") != "open":
                logger.warning(
                    f"Order {pos_id} status is not 'open' ({fresh_order_info.get('status')}). Cannot edit for TSL."
                )
                return False  # Cannot edit closed/canceled orders

            order_type = fresh_order_info.get("type")
            order_side = fresh_order_info.get("side")
            order_amount = fresh_order_info.get(
                "amount"
            )  # Use original amount? Or remaining? Check CCXT/exchange docs. Usually original amount.
            order_price = fresh_order_info.get("price")  # Required for limit orders

            if not all([order_type, order_side, order_amount]):
                logger.error(
                    f"Fetched fresh order info for {pos_id} is missing essential fields (type/side/amount). Cannot edit."
                )
                return False
            if order_type == "limit" and order_price is None:
                logger.error(f"Fetched fresh order info for limit order {pos_id} is missing price field. Cannot edit.")
                return False

            logger.debug(f"Attempting edit_order for {pos_id} with new SL: {new_tsl_price}. Params: {edit_params}")

            # Use retry decorator for the edit call itself

            @retry_api_call(max_retries=1, initial_delay=1)  # Retry edit once
            def _edit_order_attempt(exchange, order_id, sym, o_type, o_side, o_amount, o_price, params):
                return exchange.edit_order(order_id, sym, o_type, o_side, o_amount, o_price, params)

            edited_order = _edit_order_attempt(
                self.exchange, pos_id, symbol, order_type, order_side, order_amount, order_price, edit_params
            )

            if edited_order:
                confirmed_sl = edited_order.get("info", {}).get("stopLoss") or edited_order.get("stopLossPrice")
                confirmed_tp = edited_order.get("info", {}).get("takeProfit") or edited_order.get("takeProfitPrice")
                logger.info(
                    f"{Fore.MAGENTA}---> Attempted modify order {pos_id} via edit_order. Exchange Response SL: {confirmed_sl}, TP: {confirmed_tp}{Style.RESET_ALL}"
                )
                # Check if confirmed SL matches our intended TSL (allowing for string/float differences)
                if confirmed_sl and abs(float(confirmed_sl) - new_tsl_price) < 1e-9:
                    logger.info(
                        f"{Fore.GREEN}---> TSL update via edit_order appears successful for {pos_id}.{Style.RESET_ALL}"
                    )
                    return True  # Signal success to update internal state
                else:
                    logger.warning(
                        f"{Fore.YELLOW}---> TSL update via edit_order response SL ({confirmed_sl}) does not match target ({new_tsl_price}). Edit might have failed partially or confirmation is delayed.{Style.RESET_ALL}"
                    )
                    return False  # Signal potential failure
            else:
                # This case means edit_order call returned None after potential retries
                logger.error(
                    f"{Fore.RED}TSL update via edit_order for {pos_id} failed (API returned None).{Style.RESET_ALL}"
                )
                return False

        except ccxt.NotSupported as e:
            logger.error(
                f"{Fore.RED}TSL update FAILED: edit_order for SL/TP modification is NOT SUPPORTED by CCXT/{self.exchange_id} or for this order type/state. Error: {e}. Consider disabling TSL ('enable_trailing_stop_loss: false') or using a different TSL method.{Style.RESET_ALL}"
            )
            # Optionally disable TSL permanently if not supported:
            # self.enable_trailing_stop_loss = False
            # self.config['risk_management']['enable_trailing_stop_loss'] = False # Update runtime config?
            return False
        except (ccxt.OrderNotFound, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
            # These non-retryable errors caught here mean the edit failed
            logger.error(
                f"{Fore.RED}TSL update via edit_order failed for {pos_id} ({type(e).__name__}): {e}. Params: {edit_params}{Style.RESET_ALL}"
            )
            return False
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error attempting TSL order modification for {pos_id}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            return False

    def _log_position_pnl(self, position: dict, exit_price: float | None, reason: str) -> None:
        """Calculates and logs the Profit and Loss for a closed position."""
        pos_id = position["id"]
        side = position["side"]
        entry_price = position.get("entry_price")
        position_size = position.get("size")
        symbol = position.get("symbol", self.symbol)

        # Ensure all necessary values are present and valid
        if not all([pos_id, side, entry_price, position_size, exit_price, symbol]):
            logger.warning(
                f"{Fore.YELLOW}---> Position {pos_id} ({symbol}) Closed. Reason: {reason}. "
                f"Could not calculate PnL due to missing data (Entry={entry_price}, Exit={exit_price}, Size={position_size}).{Style.RESET_ALL}"
            )
            return
        if entry_price <= 0 or exit_price <= 0 or position_size <= 0:
            logger.warning(
                f"{Fore.YELLOW}---> Position {pos_id} ({symbol}) Closed. Reason: {reason}. "
                f"Could not calculate PnL due to invalid data (Entry={entry_price}, Exit={exit_price}, Size={position_size}).{Style.RESET_ALL}"
            )
            return

        try:
            # PnL calculation in quote currency
            pnl_quote = (
                (exit_price - entry_price) * position_size
                if side == "buy"
                else (entry_price - exit_price) * position_size
            )
            # PnL calculation as percentage of entry value
            pnl_pct = (
                ((exit_price / entry_price) - 1) * 100 if side == "buy" else ((entry_price / exit_price) - 1) * 100
            )

            pnl_color = Fore.GREEN if pnl_quote >= 0 else Fore.RED
            quote_ccy = self.market_info["quote"]
            base_ccy = self.market_info["base"]

            entry_price_f = f"{entry_price:.{self.price_decimals}f}"
            exit_price_f = f"{exit_price:.{self.price_decimals}f}"
            size_f = f"{position_size:.{self.amount_decimals}f}"
            pnl_quote_f = f"{pnl_quote:.{self.price_decimals}f}"
            pnl_pct_f = f"{pnl_pct:.3f}%"

            logger.info(
                f"{pnl_color}---> Position {pos_id} ({symbol}) Closed. Reason: {reason}. "
                f"Entry: {entry_price_f}, Exit: {exit_price_f}, Size: {size_f} {base_ccy}. "
                f"Est. PnL: {pnl_quote_f} {quote_ccy} ({pnl_pct_f}){Style.RESET_ALL}"
            )
            # Update simple daily PnL tracker
            self.daily_pnl += pnl_quote
            logger.info(f"Daily PnL Updated: {self.daily_pnl:.{self.price_decimals}f} {quote_ccy}")

        except ZeroDivisionError:
            logger.error(f"PnL calculation failed for {pos_id}: Division by zero (entry price was likely zero).")
        except Exception as e:
            logger.error(f"Unexpected error calculating PnL for {pos_id}: {e}", exc_info=True)

    @retry_api_call(max_retries=1)  # Retry market close once if network issue
    def _place_market_close_order(self, position: dict[str, Any], current_price: float) -> dict[str, Any] | None:
        """Places a market order to close the given position.
        Uses 'reduceOnly' parameter for safety if supported.

        Args:
            position: The position dictionary from the bot's state.
            current_price: The current market price (used for simulation & logging).

        Returns:
            The CCXT order dictionary of the close order if successful (or simulated),
            otherwise None.
        """
        pos_id = position["id"]
        side = position["side"]  # Side of the original entry order
        size = position["size"]  # Current size of the position
        symbol = position.get("symbol", self.symbol)
        base_currency = self.market_info["base"]
        self.market_info["quote"]

        if size is None or size <= 1e-12:
            logger.error(f"Cannot place market close for position {pos_id}: Size is invalid ({size}).")
            return None

        # Determine the side of the closing order (opposite of the entry side)
        close_side = "sell" if side == "buy" else "buy"
        log_color = Fore.YELLOW  # Use yellow for closing actions

        logger.warning(
            f"{log_color}Initiating MARKET CLOSE for position {pos_id} ({symbol}). "
            f"Entry Side: {side.upper()}, Close Side: {close_side.upper()}, "
            f"Size: {size:.{self.amount_decimals}f} {base_currency})...{Style.RESET_ALL}"
        )

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_close_id = f"sim_close_{int(time.time() * 1000)}_{close_side[:1]}"
            sim_avg_close_price = current_price  # Simulate fill at current price
            sim_cost = size * sim_avg_close_price
            sim_timestamp_ms = int(time.time() * 1000)
            sim_datetime = pd.to_datetime(sim_timestamp_ms, unit="ms", utc=True).isoformat()

            simulated_close_order = {
                "id": sim_close_id,
                "clientOrderId": sim_close_id,
                "timestamp": sim_timestamp_ms,
                "datetime": sim_datetime,
                "lastTradeTimestamp": sim_timestamp_ms,
                "symbol": symbol,
                "type": "market",
                "side": close_side,
                "price": None,
                "amount": size,
                "cost": sim_cost,
                "average": sim_avg_close_price,
                "filled": size,
                "remaining": 0.0,
                "status": "closed",
                "fee": None,
                "fees": [],
                "reduceOnly": True,  # Indicate it was intended as reduceOnly
                "trades": [],
                "info": {
                    "simulated": True,
                    "orderId": sim_close_id,
                    "reduceOnly": True,  # Store flag in info as well
                    "closed_position_id": pos_id,  # Link back to the position being closed
                },
            }
            logger.info(
                f"{log_color}[SIMULATION] Market Close Order Placed: ID {sim_close_id}, Size {size:.{self.amount_decimals}f}, AvgPrice {sim_avg_close_price:.{self.price_decimals}f}{Style.RESET_ALL}"
            )
            return simulated_close_order

        # --- Live Trading Mode ---
        else:
            order: dict[str, Any] | None = None
            try:
                params = {}
                # --- Use 'reduceOnly' parameter for safety ---
                # Check if exchange explicitly supports reduceOnly in createOrder params via 'has'
                # Some exchanges support it via params even if `has['reduceOnly']` is False
                reduce_only_supported = self.exchange.has.get("reduceOnly", False)  # Check explicit support flag
                if reduce_only_supported:
                    params["reduceOnly"] = True
                    logger.debug("Using 'reduceOnly=True' parameter for market close order.")
                else:
                    # Attempt sending it anyway for exchanges like Bybit V5 that use it in params
                    if self.exchange_id.lower() == "bybit":  # Example: Assume Bybit needs it in params
                        params["reduceOnly"] = True
                        logger.debug(
                            "Attempting 'reduceOnly=True' in params (exchange.has['reduceOnly'] is False/missing)."
                        )
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Exchange {self.exchange_id} might not support 'reduceOnly'. Close order might open a new position if state is wrong.{Style.RESET_ALL}"
                        )

                # Add Bybit V5 category parameter if needed
                if self.exchange_id.lower() == "bybit":
                    market_type = self.exchange.market(symbol).get("type", "swap")
                    is_linear = self.exchange.market(symbol).get("linear", True)
                    if market_type in ["swap", "future"]:
                        params["category"] = "linear" if is_linear else "inverse"
                    elif market_type == "spot":
                        params["category"] = "spot"

                # Place the market order
                logger.debug(
                    f"Placing live market close order for {pos_id}. Side: {close_side}, Size: {size}, Params: {params}"
                )
                order = self.exchange.create_market_order(symbol, close_side, size, params=params)

                if order:
                    oid = order.get("id", "N/A")
                    oavg = order.get("average")  # Actual average fill price
                    ostatus = order.get("status", STATUS_UNKNOWN)
                    ofilled = order.get("filled", 0.0)
                    ofilled_f = f"{ofilled:.{self.amount_decimals}f}" if ofilled is not None else "N/A"
                    oavg_f = f"{oavg:.{self.price_decimals}f}" if oavg is not None else "N/A (Market Fill)"

                    logger.info(
                        f"{log_color}---> LIVE Market Close Order Placed: ID {oid}, "
                        f"Status: {ostatus}, Filled: {ofilled_f}, AvgFill: {oavg_f}{Style.RESET_ALL}"
                    )
                    return order
                else:
                    # API call returned None/empty after potential retries
                    logger.error(
                        f"{Fore.RED}LIVE Market Close order placement failed: API returned None or empty response for position {pos_id}.{Style.RESET_ALL}"
                    )
                    logger.error(f" -> Params Sent: {params}")
                    return None

            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                # Specific errors often indicating state mismatch or parameter issues
                logger.error(
                    f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} ({type(e).__name__}): {e}. Position might already be closed or size/params incorrect.{Style.RESET_ALL}"
                )
                logger.error(f" -> Params Sent: {params}")
                return None
            except ccxt.ExchangeError as e:
                # General exchange errors
                err_str = str(e).lower()
                # Check for common Bybit "already closed" type errors
                if "order cost not meet" in err_str or "position size is zero" in err_str or "reduce-only" in err_str:
                    logger.warning(
                        f"{Fore.YELLOW}Market close order for {pos_id} failed, likely because position is already closed or reduce-only conflict. Error: {e}{Style.RESET_ALL}"
                    )
                    # Consider this a success in terms of the position being closed, return a dummy success? Or None?
                    # Returning None is safer, indicates the close *attempt* failed.
                    return None
                else:
                    logger.error(
                        f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} (ExchangeError): {e}{Style.RESET_ALL}"
                    )
                    logger.error(f" -> Params Sent: {params}")
                    return None
            except Exception as e:
                # Unexpected Python errors
                logger.error(
                    f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} (Unexpected Error): {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                logger.error(f" -> Params Sent: {params}")
                return None

    # --- Main Bot Execution Loop ---

    def _fetch_market_data(self) -> dict[str, Any] | None:
        """Fetches all required market data (price, order book, history) for one iteration."""
        logger.debug("Fetching market data bundle...")
        start_time = time.time()
        try:
            # Fetch concurrently? For simplicity, sequential is used.
            # Consider asyncio or threading for performance optimization if needed.
            current_price = self.fetch_market_price()
            # Order book is optional for core logic but used in signal score
            order_book_data = self.fetch_order_book()
            # History is essential for indicators
            historical_data = self.fetch_historical_data()  # Fetches required lookback automatically

            # --- Check if essential data was fetched ---
            if current_price is None:
                logger.warning(
                    f"{Fore.YELLOW}Failed to fetch current market price. Skipping iteration.{Style.RESET_ALL}"
                )
                return None
            if historical_data is None or historical_data.empty:
                # Allow running even with minimal history if fetch partially succeeded?
                # For now, require valid history.
                logger.warning(
                    f"{Fore.YELLOW}Failed to fetch sufficient historical data. Skipping iteration.{Style.RESET_ALL}"
                )
                return None

            order_book_imbalance = None
            if order_book_data is None:
                logger.debug("Order book data fetch failed or returned None. Proceeding without imbalance signal.")
            else:
                order_book_imbalance = order_book_data.get("imbalance")  # Can be None, float, or inf

            fetch_duration = time.time() - start_time
            logger.debug(f"Market data fetched successfully in {fetch_duration:.2f}s.")
            return {
                "price": current_price,
                "order_book_imbalance": order_book_imbalance,
                "historical_data": historical_data,
            }
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error during market data fetching phase: {e}{Style.RESET_ALL}", exc_info=True
            )
            return None

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> dict[str, Any]:
        """Calculates all required technical indicators based on the historical data."""
        logger.debug("Calculating technical indicators...")
        start_time = time.time()
        indicators = {}
        # Basic checks on input data
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
            logger.error("Invalid or empty historical data provided for indicator calculation.")
            return {}
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in historical_data.columns for col in required_cols):
            logger.error(
                f"Historical data DataFrame is missing required columns ({required_cols}). Cannot calculate indicators."
            )
            return {}
        if len(historical_data) < 2:
            logger.warning(f"Insufficient historical data rows ({len(historical_data)}) for most indicators.")
            # Allow calculation, individual functions will return None if needed

        # --- Safely access columns ---
        # Use .get() with default None in case columns exist but contain issues? No, checked above.
        close = historical_data["close"]
        high = historical_data["high"]
        low = historical_data["low"]

        # --- Calculate Indicators using Static Methods ---
        # Pass relevant series and periods from self.config
        indicators["volatility"] = self.calculate_volatility(close, self.volatility_window)
        indicators["ema"] = self.calculate_ema(close, self.ema_period)
        indicators["rsi"] = self.calculate_rsi(close, self.rsi_period)
        macd_tuple = self.calculate_macd(close, self.macd_short_period, self.macd_long_period, self.macd_signal_period)
        indicators["macd_line"], indicators["macd_signal"], indicators["macd_hist"] = macd_tuple
        stoch_tuple = self.calculate_stoch_rsi(
            close, self.rsi_period, self.stoch_rsi_period, self.stoch_rsi_k_period, self.stoch_rsi_d_period
        )
        indicators["stoch_k"], indicators["stoch_d"] = stoch_tuple
        indicators["atr"] = self.calculate_atr(high, low, close, self.atr_period)

        calc_duration = time.time() - start_time

        # --- Log Calculated Indicators ---
        # Helper to format indicator values for logging
        def format_ind(key: str, precision: int | None = None) -> str:
            val = indicators.get(key)
            if val is None:
                return "N/A"
            try:
                # Use specific precision if provided, else default based on type
                if precision is not None:
                    return f"{val:.{precision}f}"
                # Default formatting: 1 decimal for RSI/Stoch, price decimals for EMA/MACD/ATR
                if key in ["rsi", "stoch_k", "stoch_d"]:
                    return f"{val:.1f}"
                if key in ["ema", "atr", "macd_line", "macd_signal"]:
                    return f"{val:.{self.price_decimals}f}"
                return str(round(val, 5))  # Generic fallback
            except (TypeError, ValueError):
                return str(val)

        log_msg = (
            f"Indicators (calc took {calc_duration:.2f}s): "
            f"EMA={format_ind('ema')}, RSI={format_ind('rsi')}, ATR={format_ind('atr')}, "
            f"MACD(L/S)=({format_ind('macd_line')}/{format_ind('macd_signal')}), "
            f"Stoch(K/D)=({format_ind('stoch_k')}/{format_ind('stoch_d')}), "
            f"Vol={format_ind('volatility', 5)}"  # Volatility needs higher precision maybe
        )
        logger.info(log_msg)

        # Check for None values which indicate calculation issues
        nones = [k for k, v in indicators.items() if v is None]
        if nones:
            logger.warning(
                f"{Fore.YELLOW}Indicator calculation issue: {', '.join(nones)} resulted in None. Check data sufficiency or period settings.{Style.RESET_ALL}"
            )

        return indicators

    def _process_signals_and_entry(self, market_data: dict, indicators: dict) -> None:
        """Analyzes market data and indicators to compute a trade signal.
        If a signal meets the threshold and conditions allow (e.g., max positions,
        valid order size), attempts to place an entry order.

        Args:
            market_data: Dictionary containing current price, order book imbalance, etc.
            indicators: Dictionary containing calculated indicator values.
        """
        current_price = market_data["price"]
        orderbook_imbalance = market_data["order_book_imbalance"]
        atr_value = indicators.get("atr")  # Needed for potential ATR-based SL/TP

        # --- 1. Check Pre-conditions for Entry ---
        # Check Max Positions Limit
        active_or_pending_count = sum(
            1 for p in self.open_positions if p.get("status") in [STATUS_ACTIVE, STATUS_PENDING_ENTRY]
        )
        if active_or_pending_count >= self.max_open_positions:
            logger.info(
                f"{Fore.CYAN}Max open/pending positions ({self.max_open_positions}) reached. Skipping new entry evaluation.{Style.RESET_ALL}"
            )
            return

        # Calculate potential order size - do this early to see if trading is possible
        # This function already logs extensively and checks limits/balance.
        order_size_base = self.calculate_order_size(current_price)
        if order_size_base <= 0:
            # The reason (balance, limits, etc.) should have been logged by calculate_order_size
            logger.warning(
                f"{Fore.YELLOW}Cannot evaluate entry signal: Calculated order size is zero or invalid.{Style.RESET_ALL}"
            )
            return

        # --- 2. Compute Signal Score ---
        signal_score, reasons = self.compute_trade_signal_score(current_price, indicators, orderbook_imbalance)
        # Log score and reasons
        score_color = Fore.GREEN if signal_score > 0 else (Fore.RED if signal_score < 0 else Fore.WHITE)
        logger.info(f"Trade Signal Score: {score_color}{signal_score}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG) or signal_score != 0:  # Log reasons if debug or non-zero score
            for reason in reasons:
                logger.debug(f"  -> {reason}")  # Log reasons as debug unless score triggers action

        # --- 3. Determine Entry Action based on Threshold ---
        entry_side: str | None = None
        if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = "buy"
            logger.info(
                f"{Fore.GREEN}Buy signal score ({signal_score}) meets/exceeds threshold ({ENTRY_SIGNAL_THRESHOLD_ABS}). Preparing entry...{Style.RESET_ALL}"
            )
        elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = "sell"
            logger.info(
                f"{Fore.RED}Sell signal score ({signal_score}) meets/exceeds threshold ({-ENTRY_SIGNAL_THRESHOLD_ABS}). Preparing entry...{Style.RESET_ALL}"
            )
        else:
            # Log only if score was calculated (avoids spamming neutral messages)
            if signal_score is not None:
                logger.info(
                    f"Neutral signal score ({signal_score}). Threshold ({ENTRY_SIGNAL_THRESHOLD_ABS}) not met. No entry action."
                )
            return  # No entry signal, exit this function

        # --- 4. Place Entry Order if Signal Strong Enough ---
        if entry_side:
            # --- Calculate SL/TP Prices BEFORE placing order ---
            sl_price, tp_price = self._calculate_sl_tp_prices(
                entry_price=current_price,  # Use current price as estimate for calculation basis
                side=entry_side,
                current_price=current_price,  # Pass current price for sanity checks
                atr=atr_value,
            )

            # --- Critical Check: Abort if essential SL/TP failed when required ---
            sl_tp_required = self.use_atr_sl_tp or (self.base_stop_loss_pct is not None)
            sl_tp_failed = sl_price is None and tp_price is None  # Check if BOTH failed
            if sl_tp_required and sl_tp_failed:
                logger.error(
                    f"{Fore.RED}SL/TP calculation failed but is required by config. Cannot place entry order. Check ATR/percentage settings and logs.{Style.RESET_ALL}"
                )
                return  # Abort entry
            elif sl_tp_failed:  # SL/TP not strictly required, but calculation failed anyway
                logger.warning(
                    f"{Fore.YELLOW}SL/TP calculation failed or resulted in None (check logs). Proceeding without SL/TP parameters.{Style.RESET_ALL}"
                )

            # --- Place the Entry Order ---
            entry_order = self.place_entry_order(
                side=entry_side,
                order_size_base=order_size_base,  # Pass validated BASE currency size
                confidence_level=signal_score,
                order_type=self.entry_order_type,
                current_price=current_price,
                stop_loss_price=sl_price,  # Pass calculated SL (float or None)
                take_profit_price=tp_price,  # Pass calculated TP (float or None)
            )

            # --- Update Bot State if Order Placement Succeeded ---
            if entry_order:
                order_id = entry_order.get("id")
                if not order_id:
                    logger.error(
                        f"{Fore.RED}Entry order placed but received no order ID! Cannot track position. Manual check required! Order response: {str(entry_order)[:200]}...{Style.RESET_ALL}"
                    )
                    # Attempt to cancel if possible? Difficult without ID. Best to stop or warn heavily.
                    return  # Cannot track without ID

                # Determine initial status based on returned order status
                order_status = entry_order.get("status")
                initial_pos_status = STATUS_UNKNOWN
                is_filled_immediately = False

                if order_status == "open":
                    initial_pos_status = STATUS_PENDING_ENTRY
                    logger.info(f"Entry order {order_id} placed with status 'open'. Added as PENDING.")
                elif order_status == "closed":  # Assumed filled for market or immediate limit fills
                    initial_pos_status = STATUS_ACTIVE
                    is_filled_immediately = True
                    logger.info(
                        f"Entry order {order_id} placed with status 'closed'. Assuming FILLED. Added as ACTIVE."
                    )
                elif order_status in ["canceled", "rejected", "expired"]:
                    logger.warning(
                        f"{Fore.YELLOW}Entry order {order_id} failed immediately on placement (Status: {order_status}). Not adding position.{Style.RESET_ALL}"
                    )
                    return  # Don't add failed orders to state
                else:  # e.g., 'triggered', partial fills reported as 'open' or 'closed' depending on exchange
                    logger.warning(
                        f"Entry order {order_id} has unusual initial status: '{order_status}'. Treating as PENDING for safety."
                    )
                    initial_pos_status = STATUS_PENDING_ENTRY  # Treat unknowns as pending

                # --- Prepare Position Data for State ---
                filled_amount = float(entry_order.get("filled", 0.0))
                avg_fill_price = float(entry_order.get("average")) if entry_order.get("average") else None
                order_timestamp_ms = entry_order.get("timestamp")
                # Use requested amount for original size
                requested_amount = float(entry_order.get("amount", order_size_base))

                # Use fill price/time if known immediately, else use estimates
                entry_price_for_state = avg_fill_price if is_filled_immediately and avg_fill_price else None
                entry_time_for_state = (
                    order_timestamp_ms / 1000 if is_filled_immediately and order_timestamp_ms else None
                )
                size_for_state = filled_amount if is_filled_immediately else requested_amount

                # Create the new position dictionary
                new_position = {
                    "id": order_id,
                    "symbol": self.symbol,
                    "side": entry_side,
                    "size": size_for_state,  # Current size reflects fill status (might be 0 if pending)
                    "original_size": requested_amount,  # Always store the requested size
                    "entry_price": entry_price_for_state,  # None if pending
                    "entry_time": entry_time_for_state,  # None if pending
                    "status": initial_pos_status,
                    "entry_order_type": self.entry_order_type,
                    "stop_loss_price": sl_price,  # Store calculated SL sent with request
                    "take_profit_price": tp_price,  # Store calculated TP sent with request
                    "confidence": signal_score,
                    "trailing_stop_price": None,  # TSL not active initially
                    "last_update_time": time.time(),
                }
                self.open_positions.append(new_position)
                entry_price_f = format_price(entry_price_for_state)
                logger.info(
                    f"{Fore.CYAN}---> Position {order_id} added to state. Status: {initial_pos_status}, Entry Price: {entry_price_f}{Style.RESET_ALL}"
                )
                self._save_state()  # Save state after adding the new position
            else:
                # Order placement failed (place_entry_order returned None)
                logger.error(
                    f"{Fore.RED}Entry order placement failed (API call error or validation). No position added.{Style.RESET_ALL}"
                )
        # End of if entry_side

    def run(self) -> None:
        """Starts the main trading loop of the bot."""
        logger.info(
            f"{Fore.CYAN}--- Initiating Trading Loop (Symbol: {self.symbol}, Timeframe: {self.timeframe}) ---{Style.RESET_ALL}"
        )

        # --- Optional Initial Cleanup ---
        # logger.warning("Performing initial check/cancel of existing open orders for this symbol...")
        # try:
        #      initial_cancel_count = self.cancel_all_symbol_orders()
        #      if initial_cancel_count == -1: logger.info("Initial cancel check attempted via cancelAllOrders.")
        #      elif initial_cancel_count > 0: logger.info(f"Initial check cancelled {initial_cancel_count} open orders.")
        #      else: logger.info("No open orders found during initial check.")
        # except Exception as cancel_err:
        #      logger.error(f"Error during initial order cancellation: {cancel_err}")

        while True:
            self.iteration += 1
            start_time_iter = time.time()
            # Use UTC time for consistency
            timestamp_now = pd.Timestamp.now(tz="UTC")
            timestamp_str = timestamp_now.isoformat(timespec="seconds")
            loop_prefix = f"{Fore.BLUE}===== Iteration {self.iteration} ===={Style.RESET_ALL}"
            logger.info(f"\n{loop_prefix} Timestamp: {timestamp_str}")

            try:
                # --- 1. Fetch Market Data Bundle ---
                market_data = self._fetch_market_data()
                if market_data is None:
                    logger.warning("Essential market data missing, pausing before next iteration.")
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue

                current_price = market_data["price"]
                ob_imb = market_data.get("order_book_imbalance")
                ob_imb_str = (
                    f"{ob_imb:.3f}"
                    if isinstance(ob_imb, (float, int))
                    else ("Inf" if ob_imb == float("inf") else "N/A")
                )

                # Log current price using stored price_decimals
                logger.info(f"{loop_prefix} Current Price: {format_price(current_price)} OB Imbalance: {ob_imb_str}")

                # --- 2. Calculate Indicators ---
                indicators = self._calculate_indicators(market_data["historical_data"])
                if not indicators:  # Check if indicator calculation failed critically
                    logger.error("Indicator calculation failed critically. Skipping trading logic.")
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue

                # --- 3. Check Pending Entry Orders ---
                # Checks if limit orders were filled, updates state, recalculates SL/TP
                self._check_pending_entries(indicators)

                # --- 4. Manage Active Positions ---
                # Handles TSL updates, time-based exits, checks for external closures (SL/TP hits)
                # Run this *before* checking for new entries to free up position slots
                self._manage_active_positions(current_price, indicators)

                # --- 5. Process Signals & Potential New Entry ---
                # Evaluates indicators/market data for new trade opportunities
                self._process_signals_and_entry(market_data, indicators)

                # --- 6. Loop Pacing ---
                end_time_iter = time.time()
                execution_time = end_time_iter - start_time_iter
                wait_time = max(0.1, DEFAULT_SLEEP_INTERVAL_SECONDS - execution_time)
                logger.debug(f"{loop_prefix} Iteration took {execution_time:.2f}s. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            except KeyboardInterrupt:
                logger.warning(
                    f"\n{Fore.YELLOW}Keyboard interrupt received. Initiating graceful shutdown...{Style.RESET_ALL}"
                )
                break  # Exit the while loop to trigger finally block in main
            except SystemExit as e:
                logger.warning(f"SystemExit called with code {e.code}. Exiting main loop...")
                raise e  # Re-raise to be caught by main block's finally
            except Exception as e:
                # Catch-all for unexpected errors within the main loop
                logger.critical(
                    f"{Fore.RED}{Style.BRIGHT}{loop_prefix} CRITICAL UNHANDLED ERROR in main loop: {type(e).__name__} - {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                # Implement a cool-down period before potentially retrying
                error_pause_seconds = 60
                logger.warning(
                    f"{Fore.YELLOW}Pausing for {error_pause_seconds} seconds due to critical loop error before next attempt...{Style.RESET_ALL}"
                )
                try:
                    time.sleep(error_pause_seconds)
                except KeyboardInterrupt:
                    logger.warning(f"\n{Fore.YELLOW}Keyboard interrupt during error pause. Exiting...{Style.RESET_ALL}")
                    break  # Exit loop if interrupted during pause

        logger.info("Main trading loop terminated.")

    def shutdown(self) -> None:
        """Performs graceful shutdown procedures for the bot."""
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}--- Initiating Graceful Shutdown Sequence ---{Style.RESET_ALL}")

        # --- 1. Cancel Open PENDING Orders ---
        # Focus on orders known to the bot that haven't filled yet.
        pending_orders_to_cancel = [p for p in list(self.open_positions) if p.get("status") == STATUS_PENDING_ENTRY]
        cancelled_pending_count = 0
        if not pending_orders_to_cancel:
            logger.info("No pending entry orders found in state to cancel.")
        else:
            logger.info(f"Attempting to cancel {len(pending_orders_to_cancel)} PENDING entry order(s)...")
            for pos in pending_orders_to_cancel:
                pos_id = pos.get("id")
                pos_symbol = pos.get("symbol", self.symbol)
                if not pos_id:
                    continue

                logger.info(f"Cancelling pending order {pos_id} ({pos_symbol})...")
                if self.cancel_order_by_id(pos_id, symbol=pos_symbol):
                    cancelled_pending_count += 1
                    # Update state immediately to reflect cancellation during shutdown
                    pos["status"] = STATUS_CANCELED  # Mark as canceled in our state
                    pos["last_update_time"] = time.time()
                else:
                    logger.error(
                        f"{Fore.RED}Failed to cancel pending order {pos_id} ({pos_symbol}) during shutdown. Manual check recommended.{Style.RESET_ALL}"
                    )
            logger.info(
                f"Attempted cancellation of {len(pending_orders_to_cancel)} pending orders. Success/Already Gone: {cancelled_pending_count}."
            )
            # Save state after potential cancellations
            self._save_state()

        # --- 2. Optionally Close ACTIVE Positions ---
        active_positions_to_manage = [p for p in self.open_positions if p.get("status") == STATUS_ACTIVE]
        if self.close_positions_on_exit and active_positions_to_manage:
            logger.warning(
                f"{Fore.YELLOW}Configuration 'close_positions_on_exit' is TRUE. Attempting to close {len(active_positions_to_manage)} active position(s) via market order...{Style.RESET_ALL}"
            )

            # Fetch current price once for all closures
            current_price_for_close = self.fetch_market_price()
            if current_price_for_close is None:
                logger.critical(
                    f"{Fore.RED}{Style.BRIGHT}CRITICAL: Cannot fetch current price for market close during shutdown. {len(active_positions_to_manage)} position(s) WILL REMAIN OPEN! Manual intervention required!{Style.RESET_ALL}"
                )
            else:
                closed_count = 0
                failed_close_ids = []
                needs_state_save = False
                for position in list(active_positions_to_manage):  # Iterate copy
                    pos_id = position.get("id")
                    pos_symbol = position.get("symbol", self.symbol)
                    logger.info(f"Attempting market close for active position {pos_id} ({pos_symbol})...")
                    close_order = self._place_market_close_order(position, current_price_for_close)

                    if close_order:
                        closed_count += 1
                        logger.info(
                            f"{Fore.YELLOW}---> Market close order successfully placed for position {pos_id}. Marking as '{STATUS_CLOSED_ON_EXIT}' in state.{Style.RESET_ALL}"
                        )
                        # Update state immediately after placing close order
                        position["status"] = STATUS_CLOSED_ON_EXIT  # Custom status
                        position["last_update_time"] = time.time()
                        needs_state_save = True  # Flag that state needs saving
                        # Optionally log PnL based on estimated close price
                        close_fill_price = close_order.get("average") or current_price_for_close
                        self._log_position_pnl(
                            position, close_fill_price, f"Closed on Exit (Market Order {close_order.get('id')})"
                        )
                    else:
                        failed_close_ids.append(pos_id)
                        logger.error(
                            f"{Fore.RED}{Style.BRIGHT}CRITICAL: Failed to place market close order for {pos_id} ({pos_symbol}) during shutdown. POSITION REMAINS OPEN! Manual check required!{Style.RESET_ALL}"
                        )

                logger.info(
                    f"Attempted to close {len(active_positions_to_manage)} active positions. Market orders placed for: {closed_count}. Failed for: {len(failed_close_ids)}."
                )
                if failed_close_ids:
                    logger.error(f"Failed close order IDs: {failed_close_ids}. MANUAL ACTION REQUIRED.")
                # Save state if any positions were marked as closed
                if needs_state_save:
                    self._save_state()

        elif active_positions_to_manage:
            # Log remaining active positions if not closing them
            logger.warning(
                f"{Fore.YELLOW}{len(active_positions_to_manage)} position(s) remain active ('close_positions_on_exit' is false). Manual management may be required:{Style.RESET_ALL}"
            )
            for pos in active_positions_to_manage:
                pos_id = pos.get("id")
                side = pos.get("side")
                size = pos.get("size")
                entry = pos.get("entry_price")
                symbol = pos.get("symbol", self.symbol)
                logger.warning(
                    f" -> ID: {pos_id} ({symbol}), Side: {side}, Size: {format_amount(size)}, Entry: {format_price(entry)}"
                )
        else:
            logger.info("No active positions found in state to manage during shutdown.")

        # --- 3. Final State Save (ensure it happens even if no changes in step 1 or 2) ---
        logger.info("Saving final bot state one last time...")
        self._save_state()

        logger.info(f"{Fore.CYAN}{Style.BRIGHT}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")
        # Ensure all log handlers are flushed and closed properly
        logging.shutdown()


# --- Helper Functions (outside class for clarity, could be inside) ---
def format_price(price: float | int | str | None, decimals: int = DEFAULT_PRICE_DECIMALS) -> str:
    """Safely formats a price value to a specified number of decimal places."""
    if price is None:
        return "N/A"
    try:
        price_f = float(price)
        return f"{price_f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(price)  # Fallback to string representation


def format_amount(amount: float | int | str | None, decimals: int = DEFAULT_AMOUNT_DECIMALS) -> str:
    """Safely formats an amount value to a specified number of decimal places."""
    if amount is None:
        return "N/A"
    try:
        amount_f = float(amount)
        return f"{amount_f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(amount)  # Fallback


# --- Main Execution Block ---
if __name__ == "__main__":
    # Set thread name for main thread
    # threading.current_thread().name = "MainThread" # Already set by default, but can be explicit

    bot_instance: ScalpingBot | None = None
    exit_code: int = 0

    try:
        # --- Pre-run Setup ---
        # Ensure state directory exists if STATE_FILE_NAME includes a path
        state_dir = os.path.dirname(STATE_FILE_NAME)
        if state_dir and not os.path.exists(state_dir):
            try:
                os.makedirs(state_dir)
            except OSError:
                sys.exit(1)

        # --- Initialize and Run Bot ---
        # Initialization handles config, validation, exchange, market info, state loading
        bot_instance = ScalpingBot(config_file=CONFIG_FILE_NAME, state_file=STATE_FILE_NAME)

        # Pass formatting functions to instance if needed, or use static methods
        bot_instance.format_price = lambda p: format_price(p, bot_instance.price_decimals)
        bot_instance.format_amount = lambda a: format_amount(a, bot_instance.amount_decimals)

        bot_instance.run()  # Start the main trading loop

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully (already handled inside run, but catch here too)
        logger.warning(f"\n{Fore.YELLOW}Shutdown signal detected (KeyboardInterrupt in main block).{Style.RESET_ALL}")
        exit_code = 130  # Standard exit code for Ctrl+C

    except SystemExit as e:
        # Catch sys.exit() calls (e.g., from config validation failure)
        exit_code = e.code if isinstance(e.code, int) else 1  # Default to 1 if code is None or not int
        if exit_code not in [0, 130]:  # Log if not clean exit or Ctrl+C
            logger.error(f"Bot exited via SystemExit with unexpected code: {exit_code}.")
        else:
            logger.info(f"Bot exited via SystemExit with code: {exit_code}.")

    except Exception as e:
        # Catch any critical unhandled exceptions during setup or run
        logger.critical(
            f"{Fore.RED}{Style.BRIGHT}An unhandled critical error occurred outside the main loop: {type(e).__name__} - {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        exit_code = 1  # Indicate an error exit

    finally:
        # --- Graceful Shutdown ---
        # This block executes regardless of how the try block was exited (normal exit, exception, interrupt)
        logger.info("Initiating final shutdown procedures...")
        if bot_instance:
            try:
                # Shutdown handles cancelling pending, optionally closing active, saving state, closing logs
                bot_instance.shutdown()
            except Exception:
                # Use print as logger might be closed by shutdown()
                # Attempt to close logger handlers again just in case
                logging.shutdown()
        else:
            # If bot instance creation failed, just ensure logging is shut down
            logging.shutdown()

        sys.exit(exit_code)
