"""Scalping Bot v6.2 - Pyrmethus Enhanced Edition (Bybit V5 Optimized).

Implements an enhanced cryptocurrency scalping bot using ccxt, specifically
tailored for Bybit V5 API's parameter-based Stop Loss (SL) and Take Profit (TP)
handling. This version incorporates fixes for precision errors, improved state
management, enhanced logging, robust error handling, refined position sizing,
and clearer code structure based on prior versions and analysis.

Key Enhancements V6.2 (Compared to V6.1):
- Integrated enhanced `calculate_order_size` accepting indicators/signal_score.
- Bound config values to dedicated instance attributes in `__init__`.
- Added Bybit V5 'category' parameter more consistently in API calls.
- Refined state management (`_load_state`, `_save_state`) with more validation,
  robust atomic writes using `tempfile`, and better backup handling.
- Strengthened `_manage_active_positions`: Improved external closure detection logic,
  refined TSL update/edit logic with clearer warnings and parameter handling
  (resending TP when editing SL). Made state updates within loops safer.
- Enhanced `validate_config` for more thorough checks.
- Improved logging with more context (function names) and consistency.
- Refined indicator calculations with better minimum length checks.
- Refactored `run` method using internal helper methods for clarity.
- Improved type hinting and docstrings throughout.
- Refined shutdown procedure.
"""

import contextlib
import json
import logging
import os
import shutil  # For state file backup and management
import sys
import tempfile  # For atomic state saving
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
LOG_FILE_NAME: str = "scalping_bot_v6.log"  # Updated log file name
DEFAULT_EXCHANGE_ID: str = "bybit"
DEFAULT_TIMEFRAME: str = "1m"
DEFAULT_RETRY_MAX: int = 3
DEFAULT_RETRY_DELAY_SECONDS: int = 3
DEFAULT_SLEEP_INTERVAL_SECONDS: int = 10
STRONG_SIGNAL_THRESHOLD_ABS: int = 3  # Absolute score threshold for strong signal adjustment
ENTRY_SIGNAL_THRESHOLD_ABS: int = 2  # Absolute score threshold to trigger entry
ATR_MULTIPLIER_SL: float = 2.0       # Default ATR multiplier for Stop Loss
ATR_MULTIPLIER_TP: float = 3.0       # Default ATR multiplier for Take Profit
DEFAULT_PRICE_DECIMALS: int = 4      # Fallback price decimal places
DEFAULT_AMOUNT_DECIMALS: int = 8     # Fallback amount decimal places


# Position Status Constants
STATUS_PENDING_ENTRY: str = 'pending_entry'  # Order placed but not yet filled
STATUS_ACTIVE: str = 'active'           # Order filled, position is live
STATUS_CLOSING: str = 'closing'         # Manual close initiated (optional use)
STATUS_CANCELED: str = 'canceled'       # Order explicitly cancelled by bot or user
STATUS_REJECTED: str = 'rejected'       # Order rejected by exchange
STATUS_EXPIRED: str = 'expired'         # Order expired (e.g., timeInForce)
STATUS_CLOSED_EXT: str = 'closed_externally'  # Position closed by SL/TP/Manual action detected
STATUS_CLOSED_ON_EXIT: str = 'closed_on_exit'  # Position closed during bot shutdown
STATUS_UNKNOWN: str = 'unknown'         # Order status cannot be determined

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# --- Centralized Logger Setup ---
logger = logging.getLogger("ScalpingBotV6")  # Updated logger name
logger.setLevel(logging.DEBUG)  # Set logger to lowest level
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s"
)

# Console Handler (INFO level by default, configurable)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)  # Default console level
logger.addHandler(console_handler)

# File Handler (DEBUG level for detailed logs)
file_handler: logging.FileHandler | None = None
try:
    log_dir = os.path.dirname(LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(LOG_FILE_NAME, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)  # Log debug to file
    logger.addHandler(file_handler)
except OSError:
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
    non-retryable exchange errors. Enhanced logging.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds between retries.

    Returns:
        A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any | None:
            retries = 0
            delay = initial_delay
            instance_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else ''
            func_name = f"{instance_name}.{func.__name__}" if instance_name else func.__name__

            while retries <= max_retries:
                try:
                    logger.debug(f"API Call: {func_name} - Attempt {retries + 1}/{max_retries + 1}")
                    return func(*args, **kwargs)

                # --- Specific Retryable CCXT Errors ---
                except ccxt.RateLimitExceeded as e:
                    log_msg = f"Rate limit encountered ({func_name}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1}) Error: {e}"
                    logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                    log_msg = f"Network/Exchange issue ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                    logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")

                # --- Authentication / Permission Errors (Usually Fatal/Non-Retryable) ---
                except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
                    log_msg = f"Authentication/Permission ritual failed ({func_name}: {type(e).__name__} - {e}). Check API keys, permissions, IP whitelist. Aborting call (no retry)."
                    logger.critical(f"{Fore.RED}{log_msg}{Style.RESET_ALL}")
                    return None

                # --- Exchange Errors - Distinguish Retryable vs Non-Retryable ---
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # --- Non-Retryable / Fatal Conditions ---
                    if any(phrase in err_str for phrase in [
                        "order not found", "order does not exist", "unknown order",  # Order Life Cycle (Often expected)
                        "order already cancelled", "order was canceled", "cancel order failed",
                        "order is finished", "order hs been filled", "already closed", "order status error",
                        "cannot be modified", "position status",  # Bybit: position already closed etc.
                        "insufficient balance", "insufficient margin", "margin is insufficient",  # Funding
                        "available balance insufficient", "insufficient funds", "risk limit",  # Funding/Limits
                        "position side is not match",  # Bybit hedge mode issue?
                        "invalid order", "parameter error", "size too small", "price too low",  # Invalid Request / Limits
                        "price too high", "invalid price precision", "invalid amount precision",
                        "order cost not meet", "qty must be greater than", "must be greater than",
                        "api key is invalid", "invalid api key",  # Auth (redundant?)
                        "position mode not modified", "leverage not modified",  # Config/Setup issues
                        "reduceonly", "reduce-only",  # Position logic errors
                        "bad symbol", "invalid symbol",  # Config error
                        "account category not match",  # Bybit V5 category issue
                    ]):
                        # Use WARNING for common "not found" / "already closed", ERROR for others
                        log_level = logging.WARNING if ("not found" in err_str or "already" in err_str or "is finished" in err_str) else logging.ERROR
                        log_color = Fore.YELLOW if log_level == logging.WARNING else Fore.RED
                        log_msg = f"Non-retryable ExchangeError ({func_name}: {type(e).__name__} - {e}). Aborting call."
                        logger.log(log_level, f"{log_color}{log_msg}{Style.RESET_ALL}")
                        # Special case: Return a specific marker for InsufficientFunds?
                        # if "insufficient" in err_str: return "INSUFFICIENT_FUNDS" # Or handle in caller
                        return None  # Indicate non-retryable failure

                    # --- Potentially Retryable Exchange Errors ---
                    # Temporary glitches, server issues, nonce problems etc.
                    elif any(phrase in err_str for phrase in ["nonce", "timeout", "service unavailable", "internal error", "busy", "too many visits"]):
                        log_msg = f"Potentially transient ExchangeError ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                        logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")
                    # --- Unknown/Default Exchange Errors ---
                    else:
                        log_msg = f"Unclassified ExchangeError ({func_name}: {type(e).__name__} - {e}). Assuming transient, pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                        logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")

                # --- Catch-all for other unexpected exceptions ---
                except Exception as e:
                    log_msg = f"Unexpected Python exception during {func_name}: {type(e).__name__} - {e}. Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                    logger.error(f"{Fore.RED}{log_msg}{Style.RESET_ALL}", exc_info=True)  # Log traceback

                # --- Retry Logic ---
                if retries < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)  # Exponential backoff, capped
                    retries += 1
                else:
                    logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name}. Spell falters.{Style.RESET_ALL}")
                    return None  # Indicate failure after exhausting retries
            # Should not be reached
            return None
        return wrapper
    return decorator


# --- Core Scalping Bot Class ---
class ScalpingBot:
    """Pyrmethus Enhanced Scalping Bot v6.2. Optimized for Bybit V5 API.

    Implements scalping using technical indicators and order book analysis.
    Features robust state management, ATR/percentage SL/TP, parameter-based
    SL/TP (Bybit V5), and **EXPERIMENTAL** Trailing Stop Loss (TSL) via `edit_order`.

    **Disclaimer:** Trading cryptocurrencies involves significant risk. Use at your own risk.
    Thoroughly test in simulation/testnet modes. TSL via `edit_order` is experimental and
    requires verification on your exchange/CCXT version.
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME, state_file: str = STATE_FILE_NAME) -> None:
        """Initializes the bot: Loads config, validates, connects, loads state & market info."""
        logger.info(f"{Fore.MAGENTA}--- Pyrmethus Scalping Bot v6.2 Awakening ---{Style.RESET_ALL}")
        self.config: dict[str, Any] = {}
        self.state_file: str = state_file
        self.load_config(config_file)
        self.validate_config()  # Validate before using config values

        # --- Bind Core Attributes from Config/Environment ---
        # Credentials (prefer environment variables)
        self.api_key: str | None = os.getenv("BYBIT_API_KEY") or self.config.get("exchange", {}).get("api_key")
        self.api_secret: str | None = os.getenv("BYBIT_API_SECRET") or self.config.get("exchange", {}).get("api_secret")

        # Exchange Settings
        self.exchange_id: str = self.config["exchange"]["exchange_id"]
        self.testnet_mode: bool = self.config["exchange"]["testnet_mode"]

        # Trading Parameters
        self.symbol: str = self.config["trading"]["symbol"]
        self.timeframe: str = self.config["trading"]["timeframe"]
        self.simulation_mode: bool = self.config["trading"]["simulation_mode"]
        self.entry_order_type: str = self.config["trading"]["entry_order_type"]
        self.limit_order_entry_offset_pct_buy: float = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_entry_offset_pct_sell: float = self.config["trading"]["limit_order_offset_sell"]
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
        self.order_size_percentage: float = self.config["risk_management"]["order_size_percentage"]
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.use_atr_sl_tp: bool = self.config["risk_management"]["use_atr_sl_tp"]
        self.atr_sl_multiplier: float = self.config["risk_management"].get("atr_sl_multiplier", ATR_MULTIPLIER_SL)
        self.atr_tp_multiplier: float = self.config["risk_management"].get("atr_tp_multiplier", ATR_MULTIPLIER_TP)
        self.base_stop_loss_pct: float | None = self.config["risk_management"].get("stop_loss_percentage")
        self.base_take_profit_pct: float | None = self.config["risk_management"].get("take_profit_percentage")
        self.sl_trigger_by: str | None = self.config["risk_management"].get("sl_trigger_by")
        self.tp_trigger_by: str | None = self.config["risk_management"].get("tp_trigger_by")
        self.enable_trailing_stop_loss: bool = self.config["risk_management"]["enable_trailing_stop_loss"]
        self.trailing_stop_loss_percentage: float | None = self.config["risk_management"].get("trailing_stop_loss_percentage")
        self.time_based_exit_minutes: int | None = self.config["risk_management"].get("time_based_exit_minutes")
        self.strong_signal_adjustment_factor: float = self.config["risk_management"]["strong_signal_adjustment_factor"]
        self.weak_signal_adjustment_factor: float = self.config["risk_management"]["weak_signal_adjustment_factor"]

        # --- Internal Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0  # Simple daily PnL tracker
        # Stores active and pending positions as a list of dictionaries (See STATUS_* constants)
        # Structure documented in v6.1 remains valid.
        self.open_positions: list[dict[str, Any]] = []
        self.market_info: dict[str, Any] | None = None  # Cache for market details
        self.price_decimals: int = DEFAULT_PRICE_DECIMALS
        self.amount_decimals: int = DEFAULT_AMOUNT_DECIMALS

        # --- Setup & Initialization Steps ---
        self._configure_logging_level()
        self.exchange: ccxt.Exchange = self._initialize_exchange()
        self._load_market_info()  # Calculates decimals internally
        self._load_state()       # Load persistent state

        # --- Log Final Operating Modes ---
        sim_color = Fore.YELLOW if self.simulation_mode else Fore.CYAN
        test_color = Fore.YELLOW if self.testnet_mode else Fore.GREEN
        logger.warning(f"{sim_color}--- INTERNAL SIMULATION MODE: {self.simulation_mode} ---{Style.RESET_ALL}")
        logger.warning(f"{test_color}--- EXCHANGE TESTNET MODE: {self.testnet_mode} ---{Style.RESET_ALL}")

        if not self.simulation_mode:
            if not self.testnet_mode:
                logger.warning(f"{Fore.RED}{Style.BRIGHT}--- WARNING: LIVE TRADING ON MAINNET ACTIVE ---{Style.RESET_ALL}")
                logger.warning(f"{Fore.RED}{Style.BRIGHT}--- Ensure configuration and risk parameters are correct! ---{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}--- LIVE TRADING ON TESTNET ACTIVE ---{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}--- Running in full internal simulation mode. No real orders will be placed. ---{Style.RESET_ALL}")

        logger.info(f"{Fore.CYAN}Scalping Bot V6.2 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}")

    def _configure_logging_level(self) -> None:
        """Sets the console logging level based on the configuration file."""
        try:
            log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            console_handler.setLevel(log_level)
            logger.info(f"Console logging level enchanted to: {log_level_str}")
            logger.debug(f"Effective Bot Logger Level: {logging.getLevelName(logger.level)}")
            logger.debug(f"Effective Console Handler Level: {logging.getLevelName(console_handler.level)}")
            if file_handler:
                logger.debug(f"Effective File Handler Level: {logging.getLevelName(file_handler.level)}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error configuring logging level: {e}. Using default INFO.{Style.RESET_ALL}")
            console_handler.setLevel(logging.INFO)

    def _validate_state_entry(self, entry: Any, index: int) -> bool:
        """Validates a single entry from the loaded state file."""
        if not isinstance(entry, dict):
            logger.warning(f"State file entry #{index + 1} is not a dictionary, skipping: {str(entry)[:100]}...")
            return False

        required_keys = {'id', 'symbol', 'side', 'status', 'original_size'}
        # Optional but recommended keys: 'size', 'entry_order_type', 'last_update_time'
        # Keys required if status is ACTIVE: 'entry_price', 'entry_time', 'size'

        missing_keys = required_keys - entry.keys()
        if missing_keys:
            logger.warning(f"State file entry #{index + 1} (ID: {entry.get('id', 'N/A')}) missing required keys: {missing_keys}, skipping.")
            return False

        if entry['status'] == STATUS_ACTIVE:
            active_required = {'entry_price', 'entry_time', 'size'}
            missing_active = active_required - entry.keys()
            if missing_active:
                 logger.warning(f"State file entry #{index + 1} (ID: {entry['id']}) has status ACTIVE but missing keys: {missing_active}, skipping.")
                 return False
            if entry.get('entry_price') is None or entry.get('size') is None or entry.get('size', 0) <= 0:
                 logger.warning(f"State file entry #{index + 1} (ID: {entry['id']}) has status ACTIVE but invalid entry_price/size. Skipping.")
                 return False

        # Further type checks can be added here (e.g., check if 'size' is float/int)
        for key in ['size', 'original_size', 'entry_price', 'stop_loss_price', 'take_profit_price', 'trailing_stop_price']:
             if key in entry and entry[key] is not None and not isinstance(entry[key], (float, int)):
                 # Try conversion, but log warning if type is unexpected
                 try:
                     entry[key] = float(entry[key])
                 except (ValueError, TypeError):
                      logger.warning(f"State file entry #{index + 1} (ID: {entry['id']}) has non-numeric value for '{key}': {entry[key]}. Keeping original, but might cause issues.")
                      # Depending on strictness, could return False here

        return True

    def _load_state(self) -> None:
        """Loads bot state robustly from JSON file with backup and validation."""
        logger.info(f"Attempting to recall state from {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        initial_state_loaded = False

        # Attempt 1: Load main state file
        try:
            if os.path.exists(self.state_file):
                if os.path.getsize(self.state_file) == 0:
                    logger.warning(f"{Fore.YELLOW}State file {self.state_file} is empty. Starting fresh.{Style.RESET_ALL}")
                    self.open_positions = []
                    initial_state_loaded = True
                else:
                    with open(self.state_file, encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            logger.warning(f"{Fore.YELLOW}State file {self.state_file} contains only whitespace. Starting fresh.{Style.RESET_ALL}")
                            self.open_positions = []
                            initial_state_loaded = True
                        else:
                            saved_state_raw = json.loads(content)
                            if isinstance(saved_state_raw, list):
                                valid_positions = []
                                for i, pos_data in enumerate(saved_state_raw):
                                    if self._validate_state_entry(pos_data, i):
                                        valid_positions.append(pos_data)
                                self.open_positions = valid_positions
                                logger.info(f"{Fore.GREEN}Recalled {len(self.open_positions)} valid position(s) from state file.{Style.RESET_ALL}")
                                initial_state_loaded = True
                                # Try to remove old backup if load was successful
                                if os.path.exists(state_backup_file):
                                    try: os.remove(state_backup_file); logger.debug(f"Removed old state backup: {state_backup_file}")
                                    except OSError as remove_err: logger.warning(f"Could not remove old state backup {state_backup_file}: {remove_err}")
                            else:
                                raise ValueError(f"Invalid state format - expected a list, got {type(saved_state_raw)}.")
            else:
                logger.info(f"No prior state file found ({self.state_file}). Beginning anew.")
                self.open_positions = []
                initial_state_loaded = True

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Error decoding/validating primary state file {self.state_file}: {e}. Attempting recovery...{Style.RESET_ALL}")
            # Recovery attempt will happen below if initial_state_loaded is False

        # Attempt 2: Recovery from backup if primary load failed
        if not initial_state_loaded:
            if os.path.exists(state_backup_file):
                logger.warning(f"{Fore.YELLOW}Attempting to restore state from backup: {state_backup_file}{Style.RESET_ALL}")
                try:
                    # Backup the corrupted file before overwriting
                    corrupted_path = f"{self.state_file}.corrupted_{int(time.time())}"
                    if os.path.exists(self.state_file):
                         shutil.copy2(self.state_file, corrupted_path)
                         logger.warning(f"Backed up corrupted state file to {corrupted_path}")

                    shutil.copy2(state_backup_file, self.state_file)
                    logger.info(f"{Fore.GREEN}State restored from backup. Retrying load...{Style.RESET_ALL}")
                    # Recurse ONLY ONCE for recovery
                    return self._load_state()  # !!! CAUTION: Ensure this doesn't lead to infinite loop on backup failure

                except Exception as restore_err:
                    logger.error(f"{Fore.RED}Failed to restore state from backup {state_backup_file}: {restore_err}. Starting fresh.{Style.RESET_ALL}")
                    self.open_positions = []
            else:
                logger.warning(f"{Fore.YELLOW}No backup state file found. Corrupted primary file backed up (if possible). Starting fresh.{Style.RESET_ALL}")
                # Attempt to backup corrupted file if not already done
                if os.path.exists(self.state_file) and 'corrupted_path' not in locals():
                     try:
                         corrupted_path = f"{self.state_file}.corrupted_{int(time.time())}"
                         shutil.copy2(self.state_file, corrupted_path)
                         logger.warning(f"Backed up corrupted state file to {corrupted_path}")
                     except Exception as backup_err:
                          logger.error(f"{Fore.RED}Could not back up corrupted state file {self.state_file}: {backup_err}{Style.RESET_ALL}")
                self.open_positions = []

        # Ensure initial state file exists even if starting fresh
        if not os.path.exists(self.state_file) or os.path.getsize(self.state_file) == 0:
             self._save_state()

    def _save_state(self) -> None:
        """Saves the current bot state (list of open positions) atomically.
        Uses tempfile, os.replace, and creates a backup.
        """
        if not hasattr(self, 'open_positions'):
             logger.error("Cannot save state: 'open_positions' attribute missing.")
             return

        state_backup_file = f"{self.state_file}.bak"
        state_dir = os.path.dirname(self.state_file) or '.'  # Ensure directory exists

        # Use tempfile for secure temporary file creation in the same directory
        temp_fd, temp_path = None, None
        try:
            # Create temp file securely
            temp_fd, temp_path = tempfile.mkstemp(dir=state_dir, prefix=os.path.basename(self.state_file) + '.tmp_')
            logger.debug(f"Saving state ({len(self.open_positions)} positions) via temp file {temp_path}...")

            # Serialize State Data (ensure basic types)
            # Use default=str as fallback, but ensure numbers are numbers where possible
            state_data_to_save = json.loads(json.dumps(self.open_positions, default=str))

            # Write to Temporary File using the file descriptor
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(state_data_to_save, f, indent=4)
            temp_fd = None  # fd is now closed by the 'with' block

            # Create Backup of Current State (if exists) before replacing
            if os.path.exists(self.state_file):
                try:
                    shutil.copy2(self.state_file, state_backup_file)  # copy2 preserves metadata
                    logger.debug(f"Created state backup: {state_backup_file}")
                except Exception as backup_err:
                    logger.warning(f"Could not create state backup {state_backup_file}: {backup_err}. Proceeding cautiously.")

            # Atomically Replace the Old State File
            os.replace(temp_path, self.state_file)
            temp_path = None  # Reset temp_path as it's been moved
            logger.debug(f"State recorded successfully to {self.state_file}")

        except (OSError, TypeError) as e:
            logger.error(f"{Fore.RED}Could not scribe state to {self.state_file} (Error Type: {type(e).__name__}): {e}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}An unexpected error occurred while recording state: {e}{Style.RESET_ALL}", exc_info=True)
        finally:
            # Clean up temp file if it still exists (i.e., if error occurred before os.replace)
            if temp_fd is not None:  # If fd was opened but not closed by 'with' (e.g., error during dump)
                with contextlib.suppress(OSError): os.close(temp_fd)
            if temp_path is not None and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError as rm_err: logger.error(f"Error removing temp state file {temp_path}: {rm_err}")

    def load_config(self, config_file: str) -> None:
        """Loads configuration from YAML or creates a default."""
        try:
            with open(config_file, encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict):
                 logger.critical(f"{Fore.RED}Fatal: Config file {config_file} has invalid structure (must be a dictionary). Aborting.{Style.RESET_ALL}")
                 sys.exit(1)
            logger.info(f"{Fore.GREEN}Configuration spellbook loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.warning(f"{Fore.YELLOW}Configuration spellbook '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to create default config: {e}{Style.RESET_ALL}")
            sys.exit(1)  # Exit after attempting to create default
        except yaml.YAMLError as e:
            logger.critical(f"{Fore.RED}Fatal: Error parsing spellbook {config_file}: {e}. Check YAML syntax. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Fatal: Unexpected chaos loading configuration: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file."""
        # Default config structure (simplified for brevity, see v6.1 for full detail)
        default_config = {
            "logging": {"level": "INFO"},
            "exchange": {
                "exchange_id": DEFAULT_EXCHANGE_ID,
                "testnet_mode": True,
                # API keys via .env preferred: BYBIT_API_KEY, BYBIT_API_SECRET
            },
            "trading": {
                "symbol": "BTC/USDT:USDT",  # Example Bybit Linear Perpetual
                "timeframe": DEFAULT_TIMEFRAME,
                "simulation_mode": True,
                "entry_order_type": "limit",
                "limit_order_offset_buy": 0.0005,  # 0.05%
                "limit_order_offset_sell": 0.0005,  # 0.05%
                "close_positions_on_exit": False,
            },
            "order_book": {"depth": 10, "imbalance_threshold": 1.5},
            "indicators": {
                "volatility_window": 20, "volatility_multiplier": 0.0,
                "ema_period": 10, "rsi_period": 14,
                "macd_short_period": 12, "macd_long_period": 26, "macd_signal_period": 9,
                "stoch_rsi_period": 14, "stoch_rsi_k_period": 3, "stoch_rsi_d_period": 3,
                "atr_period": 14,
            },
            "risk_management": {
                "order_size_percentage": 0.02,  # Default 2% risk
                "max_open_positions": 1,
                "use_atr_sl_tp": True,
                "atr_sl_multiplier": ATR_MULTIPLIER_SL,
                "atr_tp_multiplier": ATR_MULTIPLIER_TP,
                "stop_loss_percentage": 0.005,  # 0.5% (used if use_atr_sl_tp is false)
                "take_profit_percentage": 0.01,  # 1.0% (used if use_atr_sl_tp is false)
                "sl_trigger_by": "MarkPrice",  # Bybit V5: MarkPrice, LastPrice, IndexPrice
                "tp_trigger_by": "MarkPrice",
                "enable_trailing_stop_loss": False,  # EXPERIMENTAL
                "trailing_stop_loss_percentage": 0.003,  # 0.3% (if TSL enabled)
                "time_based_exit_minutes": 60,  # 0 or null to disable
                "strong_signal_adjustment_factor": 1.0,  # Size multiplier for strong signals
                "weak_signal_adjustment_factor": 1.0,   # Size multiplier for weak signals (unused if score < entry threshold)
            },
        }
        # --- Environment Variable Overrides ---
        # (Example for order size percentage - add others as needed)
        env_order_pct = os.getenv("ORDER_SIZE_PERCENTAGE")
        if env_order_pct:
            try:
                override_val = float(env_order_pct)
                if 0 < override_val <= 1:
                    default_config["risk_management"]["order_size_percentage"] = override_val
                    logger.info(f"Overrode order_size_percentage from environment to {override_val}.")
                else: logger.warning("Invalid ORDER_SIZE_PERCENTAGE in env, using default.")
            except ValueError: logger.warning("Invalid ORDER_SIZE_PERCENTAGE in env, using default.")

        try:
            config_dir = os.path.dirname(config_file)
            if config_dir and not os.path.exists(config_dir):
                 os.makedirs(config_dir)
                 logger.info(f"Created config directory: {config_dir}")

            with open(config_file, "w", encoding='utf-8') as f:
                yaml.dump(default_config, f, indent=4, sort_keys=False, default_flow_style=False)
            logger.info(f"{Fore.YELLOW}A default spellbook has been crafted: '{config_file}'.{Style.RESET_ALL}")
            logger.info(f"{Fore.YELLOW}IMPORTANT: Please review and tailor its enchantments (especially API keys in .env or config, symbol, risk settings), then restart the bot.{Style.RESET_ALL}")
        except OSError as e:
            logger.error(f"{Fore.RED}Could not scribe default spellbook {config_file}: {e}{Style.RESET_ALL}")
            raise  # Re-raise to be caught by calling function

    def validate_config(self) -> None:
        """Performs detailed validation of the loaded configuration parameters."""
        logger.debug("Scrutinizing the configuration spellbook...")
        try:
            # Helper functions for validation
            def _get_nested(data: dict, keys: list[str], section: str, default: Any = KeyError, req_type: type | tuple[type, ...] | None = None, allow_none: bool = False):
                value = data
                full_key_path = section
                for key in keys:
                    full_key_path += f".{key}"
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        if default is KeyError: raise KeyError(f"Missing required key: '{full_key_path}'")
                        return default  # Return default if specified and key missing
                # Type check
                if value is None and allow_none: return None
                if value is None and not allow_none: raise TypeError(f"Key '{full_key_path}' cannot be None.")
                if req_type is not None and not isinstance(value, req_type):
                     # Special case: allow int where float is expected
                     if isinstance(req_type, tuple) and float in req_type and isinstance(value, int):
                          return float(value)
                     if req_type is float and isinstance(value, int):
                          return float(value)
                     raise TypeError(f"Key '{full_key_path}' expects type {req_type} but got {type(value)}.")
                return value

            def _check_range(value: int | float, key_path: str, min_val: int | float | None = None, max_val: int | float | None = None, min_exclusive: bool = False, max_exclusive: bool = False) -> None:
                if min_val is not None:
                    if min_exclusive and value <= min_val: raise ValueError(f"{key_path} ({value}) must be > {min_val}")
                    if not min_exclusive and value < min_val: raise ValueError(f"{key_path} ({value}) must be >= {min_val}")
                if max_val is not None:
                    if max_exclusive and value >= max_val: raise ValueError(f"{key_path} ({value}) must be < {max_val}")
                    if not max_exclusive and value > max_val: raise ValueError(f"{key_path} ({value}) must be <= {max_val}")

            # --- Validate Sections ---
            required_sections = ["logging", "exchange", "trading", "order_book", "indicators", "risk_management"]
            for section in required_sections:
                 if section not in self.config or not isinstance(self.config[section], dict):
                      raise ValueError(f"Config Error: Missing or invalid section '{section}'. Must be a dictionary.")

            # --- Logging ---
            log_level = _get_nested(self.config, ["level"], "logging", default="INFO", req_type=str).upper()
            if not hasattr(logging, log_level): raise ValueError(f"logging.level: Invalid level '{log_level}'.")

            # --- Exchange ---
            ex_id = _get_nested(self.config, ["exchange_id"], "exchange", req_type=str)
            if not ex_id or ex_id not in ccxt.exchanges: raise ValueError(f"exchange_id: Invalid or unsupported '{ex_id}'.")
            _get_nested(self.config, ["testnet_mode"], "exchange", req_type=bool)

            # --- Trading ---
            _get_nested(self.config, ["symbol"], "trading", req_type=str)
            _get_nested(self.config, ["timeframe"], "trading", req_type=str)
            _get_nested(self.config, ["simulation_mode"], "trading", req_type=bool)
            entry_type = _get_nested(self.config, ["entry_order_type"], "trading", req_type=str).lower()
            if entry_type not in ["market", "limit"]: raise ValueError("trading.entry_order_type: Must be 'market' or 'limit'.")
            _check_range(_get_nested(self.config, ["limit_order_offset_buy"], "trading", req_type=(float, int)), "trading.limit_order_offset_buy", min_val=0)
            _check_range(_get_nested(self.config, ["limit_order_offset_sell"], "trading", req_type=(float, int)), "trading.limit_order_offset_sell", min_val=0)
            _get_nested(self.config, ["close_positions_on_exit"], "trading", req_type=bool)

            # --- Order Book ---
            _check_range(_get_nested(self.config, ["depth"], "order_book", req_type=int), "order_book.depth", min_val=1)
            _check_range(_get_nested(self.config, ["imbalance_threshold"], "order_book", req_type=(float, int)), "order_book.imbalance_threshold", min_val=0, min_exclusive=True)

            # --- Indicators ---
            periods = ["volatility_window", "ema_period", "rsi_period", "macd_short_period",
                       "macd_long_period", "macd_signal_period", "stoch_rsi_period",
                       "stoch_rsi_k_period", "stoch_rsi_d_period", "atr_period"]
            for p in periods: _check_range(_get_nested(self.config, [p], "indicators", req_type=int), f"indicators.{p}", min_val=1)
            _check_range(_get_nested(self.config, ["volatility_multiplier"], "indicators", req_type=(float, int)), "indicators.volatility_multiplier", min_val=0)
            if self.config["indicators"]["macd_short_period"] >= self.config["indicators"]["macd_long_period"]:
                 raise ValueError("indicators: macd_short_period must be less than macd_long_period.")

            # --- Risk Management ---
            use_atr = _get_nested(self.config, ["use_atr_sl_tp"], "risk_management", req_type=bool)
            if use_atr:
                _check_range(_get_nested(self.config, ["atr_sl_multiplier"], "risk_management", req_type=(float, int)), "risk_management.atr_sl_multiplier", min_val=0, min_exclusive=True)
                _check_range(_get_nested(self.config, ["atr_tp_multiplier"], "risk_management", req_type=(float, int)), "risk_management.atr_tp_multiplier", min_val=0, min_exclusive=True)
            else:
                sl_pct = _get_nested(self.config, ["stop_loss_percentage"], "risk_management", req_type=(float, int))
                tp_pct = _get_nested(self.config, ["take_profit_percentage"], "risk_management", req_type=(float, int))
                _check_range(sl_pct, "risk_management.stop_loss_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)
                _check_range(tp_pct, "risk_management.take_profit_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)

            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None]
            sl_trig = _get_nested(self.config, ["sl_trigger_by"], "risk_management", allow_none=True)
            tp_trig = _get_nested(self.config, ["tp_trigger_by"], "risk_management", allow_none=True)
            if sl_trig not in valid_triggers: raise ValueError(f"risk_management.sl_trigger_by: Invalid trigger '{sl_trig}'.")
            if tp_trig not in valid_triggers: raise ValueError(f"risk_management.tp_trigger_by: Invalid trigger '{tp_trig}'.")

            enable_tsl = _get_nested(self.config, ["enable_trailing_stop_loss"], "risk_management", req_type=bool)
            if enable_tsl:
                tsl_pct = _get_nested(self.config, ["trailing_stop_loss_percentage"], "risk_management", req_type=(float, int))
                _check_range(tsl_pct, "risk_management.trailing_stop_loss_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)

            order_size_pct = _get_nested(self.config, ["order_size_percentage"], "risk_management", req_type=(float, int))
            _check_range(order_size_pct, "risk_management.order_size_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=False)  # <= 1
            _check_range(_get_nested(self.config, ["max_open_positions"], "risk_management", req_type=int), "risk_management.max_open_positions", min_val=1)
            time_exit = _get_nested(self.config, ["time_based_exit_minutes"], "risk_management", req_type=int, allow_none=True)
            if time_exit is not None: _check_range(time_exit, "risk_management.time_based_exit_minutes", min_val=0)
            _check_range(_get_nested(self.config, ["strong_signal_adjustment_factor"], "risk_management", req_type=(float, int)), "risk_management.strong_signal_adjustment_factor", min_val=0, min_exclusive=True)
            _check_range(_get_nested(self.config, ["weak_signal_adjustment_factor"], "risk_management", req_type=(float, int)), "risk_management.weak_signal_adjustment_factor", min_val=0, min_exclusive=True)

            logger.info(f"{Fore.GREEN}Configuration spellbook deemed valid and potent.{Style.RESET_ALL}")

        except (ValueError, TypeError, KeyError) as e:
            logger.critical(f"{Fore.RED}Configuration flaw detected: {e}. Mend the '{CONFIG_FILE_NAME}' scroll. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Unexpected chaos during configuration validation: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and configures the CCXT exchange instance."""
        logger.info(f"Summoning exchange spirits for {self.exchange_id.upper()}...")

        creds_found = self.api_key and self.api_secret
        if not self.simulation_mode and not creds_found:
             logger.critical(f"{Fore.RED}API Key/Secret essence missing (check .env or config). Cannot trade live/testnet. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        elif creds_found and self.simulation_mode:
             logger.warning(f"{Fore.YELLOW}API Key/Secret found, but internal simulation_mode is True. Credentials will NOT be used for placing orders.{Style.RESET_ALL}")
        elif not creds_found and self.simulation_mode:
             logger.info(f"{Fore.CYAN}Running in internal simulation mode. API credentials not required/found.{Style.RESET_ALL}")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap' if 'bybit' in self.exchange_id.lower() else 'future',
                    'adjustForTimeDifference': True,
                    # Add Bybit V5 default category if desired, though better handled per call
                    # 'defaultRecvWindow': 10000, # Example: if timeouts occur
                }
            }
            # Add credentials ONLY if NOT in internal simulation mode
            if not self.simulation_mode:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
                logger.debug("API credentials loaded into exchange config.")

            exchange = exchange_class(exchange_config)

            # Set testnet mode using CCXT unified method (if not simulating)
            if not self.simulation_mode and self.testnet_mode:
                logger.info("Attempting to enable exchange sandbox mode...")
                try:
                    exchange.set_sandbox_mode(True)
                    logger.info(f"{Fore.YELLOW}Exchange sandbox mode explicitly enabled via CCXT.{Style.RESET_ALL}")
                except ccxt.NotSupported:
                    logger.warning(f"{Fore.YELLOW}{self.exchange_id} may not support unified set_sandbox_mode. Testnet depends on API key type/URL.{Style.RESET_ALL}")
                except Exception as sandbox_err:
                    logger.error(f"{Fore.RED}Error setting sandbox mode: {sandbox_err}. Continuing, but testnet may not be active.{Style.RESET_ALL}")
            elif not self.simulation_mode and not self.testnet_mode:
                 logger.info(f"{Fore.GREEN}Exchange sandbox mode is OFF. Operating on Mainnet.{Style.RESET_ALL}")

            # Load markets (needed for symbol validation, precision, etc.)
            logger.debug("Loading market matrix...")
            exchange.load_markets(reload=True)
            logger.debug("Market matrix loaded.")

            # Validate timeframe and symbol against loaded data
            if self.timeframe not in exchange.timeframes:
                available_tf = list(exchange.timeframes.keys())[:15]
                logger.critical(f"{Fore.RED}Timeframe '{self.timeframe}' unsupported by {self.exchange_id}. Available: {available_tf}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)
            if self.symbol not in exchange.markets:
                available_sym = list(exchange.markets.keys())[:10]
                logger.critical(f"{Fore.RED}Symbol '{self.symbol}' not found on {self.exchange_id}. Available: {available_sym}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)

            market_details = exchange.market(self.symbol)
            if not market_details.get('active', True):
                 logger.warning(f"{Fore.YELLOW}Warning: Market '{self.symbol}' is marked as inactive.{Style.RESET_ALL}")

            logger.info(f"Symbol '{self.symbol}' and timeframe '{self.timeframe}' confirmed available.")

            # Perform initial API connectivity test (if not simulating)
            if not self.simulation_mode:
                logger.debug("Performing initial API connectivity test (fetch balance)...")
                # Use fetch_balance as a more comprehensive initial test than fetch_time
                initial_balance_check = self.fetch_balance(currency_code=market_details.get('quote'))
                if initial_balance_check is not None:  # fetch_balance returns float or 0.0 on error
                     logger.info(f"{Fore.GREEN}API connection and authentication successful (Balance Check OK).{Style.RESET_ALL}")
                else:
                     # Fetch_balance logs specific errors, just add a critical failure message here
                     logger.critical(f"{Fore.RED}Initial API connectivity/auth test (fetch_balance) failed. Check logs. Aborting.{Style.RESET_ALL}")
                     sys.exit(1)

            logger.info(f"{Fore.GREEN}Exchange spirits aligned for {self.exchange_id.upper()}.{Style.RESET_ALL}")
            return exchange

        except ccxt.AuthenticationError as e:
             logger.critical(f"{Fore.RED}Authentication failed for {self.exchange_id}: {e}. Check API keys/permissions. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
             logger.critical(f"{Fore.RED}Connection failed to {self.exchange_id} ({type(e).__name__}): {e}. Check network/status. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Unexpected error initializing exchange {self.exchange_id}: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _calculate_decimal_places(self, tick_size: float | int | str | None) -> int:
        """Calculates decimal places from tick size (e.g., 0.01 -> 2)."""
        fallback_decimals = DEFAULT_PRICE_DECIMALS  # Use a reasonable fallback
        if tick_size is None: return fallback_decimals
        try:
            # Convert string tick size to float if necessary
            if isinstance(tick_size, str): tick_size = float(tick_size)
            if not isinstance(tick_size, (float, int)) or tick_size <= 0: return fallback_decimals

            if isinstance(tick_size, int) or tick_size.is_integer(): return 0  # Integer tick size means 0 decimals

            # Use string formatting method for robustness with various float representations
            s = format(tick_size, '.16f').rstrip('0')  # Format with high precision and remove trailing zeros
            if '.' in s:
                decimals = len(s.split('.')[-1])
                # logger.debug(f"Calculated {decimals} decimals for tick size {tick_size} from string '{s}'")
                return decimals
            else:
                return 0  # Should be caught by integer check, but handle anyway
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse tick size '{tick_size}' for decimal calculation: {e}. Using fallback {fallback_decimals}.")
            return fallback_decimals
        except Exception as e:
            logger.error(f"Unexpected error calculating decimals for tick size {tick_size}: {e}. Using fallback {fallback_decimals}.")
            return fallback_decimals

    def _load_market_info(self) -> None:
        """Loads and caches market info (precision, limits), calculating decimal places."""
        logger.debug(f"Loading market details for {self.symbol}...")
        try:
            self.market_info = self.exchange.market(self.symbol)
            if not self.market_info:
                raise ValueError(f"Market info for '{self.symbol}' not found after loading markets.")

            precision = self.market_info.get('precision', {})
            limits = self.market_info.get('limits', {})
            amount_tick = precision.get('amount')
            price_tick = precision.get('price')
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')

            self.price_decimals = self._calculate_decimal_places(price_tick)
            self.amount_decimals = self._calculate_decimal_places(amount_tick)

            # Log warnings if essential info missing or calculation defaulted
            if amount_tick is None: logger.warning(f"{Fore.YELLOW}Amount tick size missing for {self.symbol}. Using default amount decimals: {self.amount_decimals}.{Style.RESET_ALL}")
            if price_tick is None: logger.warning(f"{Fore.YELLOW}Price tick size missing for {self.symbol}. Using default price decimals: {self.price_decimals}.{Style.RESET_ALL}")
            if min_amount is None: logger.warning(f"{Fore.YELLOW}Min order amount limit missing for {self.symbol}.{Style.RESET_ALL}")
            if min_cost is None: logger.warning(f"{Fore.YELLOW}Min order cost limit missing for {self.symbol}.{Style.RESET_ALL}")

            logger.info(f"Market Details for {self.symbol}: Price Decimals={self.price_decimals} (Tick: {price_tick}), Amount Decimals={self.amount_decimals} (Tick: {amount_tick})")
            logger.debug(f"Min Amount: {min_amount}, Min Cost: {min_cost}")
            logger.debug("Market details loaded and cached.")

        except (KeyError, ValueError, TypeError) as e:
             logger.critical(f"{Fore.RED}Error loading/parsing market info for {self.symbol}: {e}. Market Info: {self.market_info}. Aborting.{Style.RESET_ALL}", exc_info=False)
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Failed to load crucial market info for {self.symbol}: {e}. Aborting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    # --- Data Fetching Methods ---

    @retry_api_call()
    def fetch_market_price(self) -> float | None:
        """Fetches the last traded price for the symbol."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        last_price = ticker.get('last') if ticker else None
        if last_price is not None:
            try:
                price = float(last_price)
                logger.debug(f"Current market price ({self.symbol}): {self.format_price(price)}")
                return price
            except (ValueError, TypeError) as e:
                 logger.error(f"{Fore.RED}Error converting ticker 'last' price ({last_price}) to float for {self.symbol}: {e}{Style.RESET_ALL}")
                 return None
        else:
            logger.warning(f"{Fore.YELLOW}Could not fetch valid 'last' price for {self.symbol}. Ticker: {ticker}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_order_book(self) -> dict[str, Any] | None:
        """Fetches order book and calculates volume imbalance."""
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            result = {'bids': bids, 'asks': asks, 'imbalance': None}

            valid_bids = [bid for bid in bids if isinstance(bid, (list, tuple)) and len(bid) >= 2 and isinstance(bid[1], (int, float)) and bid[1] >= 0]
            valid_asks = [ask for ask in asks if isinstance(ask, (list, tuple)) and len(ask) >= 2 and isinstance(ask[1], (int, float)) and ask[1] >= 0]

            if not valid_bids or not valid_asks:
                 logger.debug(f"Order book data invalid/incomplete for {self.symbol}. Bids: {len(valid_bids)}/{len(bids)}, Asks: {len(valid_asks)}/{len(asks)}")
                 return result  # Return structure but imbalance will be None

            bid_volume = sum(float(bid[1]) for bid in valid_bids)
            ask_volume = sum(float(ask[1]) for ask in valid_asks)
            epsilon = 1e-12

            if bid_volume > epsilon:
                imbalance_ratio = ask_volume / bid_volume
                result['imbalance'] = imbalance_ratio
                logger.debug(f"Order Book ({self.symbol}) Imbalance (Ask/Bid): {imbalance_ratio:.3f}")
            elif ask_volume > epsilon:  # Bids near zero, asks exist
                result['imbalance'] = float('inf')
                logger.debug(f"Order Book ({self.symbol}) Imbalance: Inf (Near-Zero Bid Vol)")
            else:  # Both near zero
                result['imbalance'] = None
                logger.debug(f"Order Book ({self.symbol}) Imbalance: N/A (Near-Zero Volumes)")
            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching/processing order book for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=False)
            return None

    @retry_api_call(max_retries=2, initial_delay=1)
    def fetch_historical_data(self, limit: int | None = None) -> pd.DataFrame | None:
        """Fetches historical OHLCV data, calculating required limit based on indicators."""
        if limit is None:
             required_periods = [
                 self.volatility_window, self.ema_period, self.rsi_period + 1,
                 self.macd_long_period + self.macd_signal_period,
                 # StochRSI needs RSI series first, then Stoch window, then smoothing
                 self.rsi_period + self.stoch_rsi_period + max(self.stoch_rsi_k_period, self.stoch_rsi_d_period),
                 self.atr_period + 1
             ]
             valid_periods = [p for p in required_periods if isinstance(p, int) and p > 0]
             max_lookback = max(valid_periods) if valid_periods else 50  # Default lookback if no valid periods
             required_limit = max(max_lookback + 50, 100)  # Add buffer, ensure minimum fetch
             required_limit = min(required_limit, 1000)  # Cap request size
             fetch_limit = required_limit
             min_required_len_for_calc = max_lookback  # Min valid rows needed AFTER cleaning
        else:
             fetch_limit = limit
             min_required_len_for_calc = limit - 50  # Estimate minimum needed if limit overridden

        logger.debug(f"Fetching ~{fetch_limit} historical candles for {self.symbol} ({self.timeframe})...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=fetch_limit)
            if not ohlcv:
                logger.warning(f"{Fore.YELLOW}No historical OHLCV data returned for {self.symbol} ({self.timeframe}, limit {fetch_limit}).{Style.RESET_ALL}")
                return None
            if len(ohlcv) < 5:
                 logger.warning(f"{Fore.YELLOW}Very few historical OHLCV data points returned ({len(ohlcv)}). Insufficient.{Style.RESET_ALL}")
                 return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            initial_len = len(df)
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)  # Drop rows with NaN prices
            final_len = len(df)

            if final_len < initial_len:
                logger.debug(f"Dropped {initial_len - final_len} rows with NaNs in OHLC from history.")

            if final_len < min_required_len_for_calc:
                 logger.warning(f"{Fore.YELLOW}Insufficient valid historical data after cleaning for {self.symbol}. Got {final_len} rows, need ~{min_required_len_for_calc} for full indicator calculation. Indicators might be inaccurate.{Style.RESET_ALL}")
                 if final_len == 0: return None  # Cannot proceed if empty

            logger.debug(f"Fetched and processed {final_len} valid historical candles for {self.symbol}.")
            return df

        except ccxt.BadSymbol as e:
             logger.critical(f"{Fore.RED}BadSymbol fetching history for {self.symbol}: {e}. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching/processing historical data for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    @retry_api_call()
    def fetch_balance(self, currency_code: str | None = None) -> float | None:
        """Fetches the available balance for a specific currency (quote currency by default).
        Handles Bybit V5 Unified/Contract/Spot accounts. Returns None on failure.
        """
        func_name = "fetch_balance"
        quote_currency = currency_code or self.market_info.get('quote')
        if not quote_currency:
             logger.error(f"{Fore.RED}[{func_name}] Cannot determine quote currency. Market Info: {self.market_info}.{Style.RESET_ALL}")
             return None  # Indicate failure explicitly

        logger.debug(f"[{func_name}] Fetching available balance for {quote_currency}...")

        if self.simulation_mode:
            dummy_balance = 10000.0
            logger.warning(f"{Fore.YELLOW}[SIMULATION][{func_name}] Returning dummy balance: {self.format_price(dummy_balance)} {quote_currency}{Style.RESET_ALL}")
            return dummy_balance

        try:
            params = {}
            # Bybit V5 requires accountType hint sometimes, though ccxt might handle it
            if self.exchange_id.lower() == 'bybit':
                # Determine likely account type based on market type
                market_type = self.exchange.market(self.symbol).get('type', 'swap')
                if market_type in ['swap', 'future']:
                    # Could be UNIFIED or CONTRACT, unified is more common now.
                    # Let CCXT try default first, add param if needed later based on errors.
                    # params = {'accountType': 'UNIFIED'} # Or CONTRACT
                    pass  # Keep params empty initially
                elif market_type == 'spot':
                    params = {'accountType': 'SPOT'}

            balance_data = self.exchange.fetch_balance(params=params)
            available_balance_str: str | None = None  # Store as string initially

            # --- Strategy: Standard 'free' -> Bybit 'info' -> Fallback 'total' ---

            # 1. Standard CCXT 'free'
            if quote_currency in balance_data and balance_data[quote_currency].get('free') is not None:
                available_balance_str = str(balance_data[quote_currency]['free'])
                logger.debug(f"[{func_name}] Found standard 'free' balance: {available_balance_str}")

            # 2. Parse 'info' (Exchange-Specific - Bybit Example)
            is_free_zero = True
            with contextlib.suppress(ValueError, TypeError): is_free_zero = (available_balance_str is None or float(available_balance_str) < 1e-12)

            if is_free_zero and self.exchange_id.lower() == 'bybit':
                logger.debug(f"[{func_name}] Standard 'free' zero/missing. Checking Bybit 'info'...")
                info_data = balance_data.get('info', {})
                try:
                    # Bybit V5 /v5/account/wallet-balance structure
                    result_list = info_data.get('result', {}).get('list', [])
                    if result_list and isinstance(result_list[0], dict):
                        account_info = result_list[0]
                        account_type = account_info.get('accountType')
                        logger.debug(f"[{func_name}] Parsing Bybit 'info': Account Type = {account_type}")

                        # Unified/Contract (Margin): Prefer availableToWithdraw or availableBalance
                        if account_type in ['UNIFIED', 'CONTRACT']:
                            coin_list = account_info.get('coin', [])
                            for coin in coin_list:
                                if isinstance(coin, dict) and coin.get('coin') == quote_currency:
                                    # Prioritize availableToWithdraw, fallback to availableBalance or walletBalance
                                    avail_str = coin.get('availableToWithdraw')
                                    if avail_str is None or float(avail_str) < 1e-12:
                                         avail_str = coin.get('availableBalance')  # May include borrow? check docs
                                    if avail_str is None or float(avail_str) < 1e-12:
                                         avail_str = coin.get('walletBalance')  # Less ideal, might include unrealized PNL

                                    if avail_str is not None:
                                        available_balance_str = str(avail_str)
                                        logger.debug(f"[{func_name}] Using Bybit {account_type} coin balance ({quote_currency}): {available_balance_str}")
                                        break
                        # Spot: Prefer availableToWithdraw (free), fallback walletBalance
                        elif account_type == 'SPOT':
                            coin_list = account_info.get('coin', [])
                            for coin in coin_list:
                                if isinstance(coin, dict) and coin.get('coin') == quote_currency:
                                    avail_str = coin.get('availableToWithdraw')  # Usually 'free' for spot
                                    if avail_str is None or float(avail_str) < 1e-12:
                                         avail_str = coin.get('walletBalance')

                                    if avail_str is not None:
                                        available_balance_str = str(avail_str)
                                        logger.debug(f"[{func_name}] Using Bybit SPOT coin balance ({quote_currency}): {available_balance_str}")
                                        break
                    else: logger.debug(f"[{func_name}] Bybit 'info' structure not as expected.")
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Could not parse Bybit balance 'info': {e}. Raw: {str(info_data)[:200]}...{Style.RESET_ALL}")

            # 3. Fallback to 'total' balance (Less Accurate)
            is_available_zero = True
            with contextlib.suppress(ValueError, TypeError): is_available_zero = (available_balance_str is None or float(available_balance_str) < 1e-12)

            if is_available_zero and quote_currency in balance_data and balance_data[quote_currency].get('total') is not None:
                 total_balance_str = str(balance_data[quote_currency]['total'])
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Available balance zero/missing. Using 'total' ({total_balance_str}) as fallback for {quote_currency}. (Includes used margin/collateral){Style.RESET_ALL}")
                 available_balance_str = total_balance_str

            # --- Final Conversion and Return ---
            if available_balance_str is not None:
                try:
                    final_balance = float(available_balance_str)
                    if final_balance < 0:
                        logger.warning(f"{Fore.YELLOW}[{func_name}] Fetched balance for {quote_currency} is negative ({final_balance}). Treating as 0.0.{Style.RESET_ALL}")
                        final_balance = 0.0
                    logger.info(f"[{func_name}] Fetched available balance: {self.format_price(final_balance)} {quote_currency}")
                    return final_balance
                except (ValueError, TypeError):
                    logger.error(f"{Fore.RED}[{func_name}] Could not convert final balance '{available_balance_str}' to float for {quote_currency}. Returning None.{Style.RESET_ALL}")
                    return None
            else:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Could not determine available balance for {quote_currency}. Returning None.{Style.RESET_ALL}")
                return None  # Explicitly return None on failure

        except ccxt.AuthenticationError as e:
             logger.error(f"{Fore.RED}[{func_name}] Authentication failed fetching balance: {e}.{Style.RESET_ALL}")
             return None
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    @retry_api_call(max_retries=1)
    def fetch_order_status(self, order_id: str, symbol: str | None = None) -> dict[str, Any] | None:
        """Fetches status of a specific order. Returns None if not found or error."""
        func_name = "fetch_order_status"
        if not order_id: logger.warning(f"[{func_name}] Called with empty order_id."); return None

        target_symbol = symbol or self.symbol
        logger.debug(f"[{func_name}] Fetching status for order {order_id} ({target_symbol})...")

        # --- Simulation Mode ---
        if self.simulation_mode:
            for pos in self.open_positions:
                 if pos['id'] == order_id:
                    logger.debug(f"[SIMULATION][{func_name}] Returning cached status for order {order_id}.")
                    # Construct basic simulated order mimicking ccxt (simplified)
                    sim_status = pos.get('status', STATUS_UNKNOWN)
                    ccxt_status = 'open' if sim_status == STATUS_PENDING_ENTRY else \
                                  'closed' if sim_status == STATUS_ACTIVE else \
                                  'canceled' if sim_status == STATUS_CANCELED else \
                                  sim_status  # Pass others like 'unknown', 'rejected'
                    sim_avg = pos.get('entry_price') if sim_status == STATUS_ACTIVE else None
                    sim_filled = pos.get('size') if sim_status == STATUS_ACTIVE else 0.0
                    sim_amount = pos.get('original_size', 0.0)
                    sim_timestamp = int(pos.get('last_update_time', time.time()) * 1000)

                    return {
                        'id': order_id, 'symbol': target_symbol, 'status': ccxt_status,
                        'type': pos.get('entry_order_type', 'limit'), 'side': pos.get('side'),
                        'amount': sim_amount, 'filled': sim_filled, 'remaining': max(0, sim_amount - sim_filled),
                        'average': sim_avg, 'timestamp': sim_timestamp, 'datetime': pd.to_datetime(sim_timestamp, unit='ms', utc=True).isoformat(),
                        'stopLossPrice': pos.get('stop_loss_price'), 'takeProfitPrice': pos.get('take_profit_price'),
                        'info': {'simulated': True, 'orderId': order_id, 'internalStatus': sim_status}
                    }
            logger.warning(f"[SIMULATION][{func_name}] Simulated order {order_id} not found in state.")
            return None  # Not found

        # --- Live Mode ---
        else:
            try:
                params = {}
                # Add Bybit V5 category param hint
                if self.exchange_id.lower() == 'bybit':
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    is_linear = self.exchange.market(target_symbol).get('linear', True)  # Default to linear if unsure
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'

                order_info = self.exchange.fetch_order(order_id, target_symbol, params=params)

                status = order_info.get('status', STATUS_UNKNOWN)
                filled = order_info.get('filled', 0.0)
                avg_price = order_info.get('average')
                logger.debug(f"[{func_name}] Order {order_id}: Status={status}, Filled={self.format_amount(filled)}, AvgPrice={self.format_price(avg_price)}")
                return order_info

            except ccxt.OrderNotFound as e:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} ({target_symbol}) not found. Assumed closed/cancelled/invalid. Error: {e}{Style.RESET_ALL}")
                return None
            except ccxt.ExchangeError as e:
                 err_str = str(e).lower()
                 # Treat specific errors as "not found" or final state
                 if any(phrase in err_str for phrase in ["order is finished", "order has been filled", "already closed", "order status error"]):
                      logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} reported as finished/filled/closed. Assuming final state. Error: {e}{Style.RESET_ALL}")
                      return None
                 else:
                      logger.error(f"{Fore.RED}[{func_name}] Exchange error fetching order {order_id}: {e}. Returning None.{Style.RESET_ALL}")
                      return None
            except Exception as e:
                logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching order {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
                return None

    # --- Indicator Calculation Methods (Static) ---
    # (Includes basic checks for data length)
    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> float | None:
        if close_prices is None or window <= 0 or len(close_prices) < window + 1: return None
        try:
            if (close_prices <= 0).any(): return None  # Log returns require positive prices
            log_returns = np.log(close_prices / close_prices.shift(1))
            volatility = log_returns.rolling(window=window, min_periods=window).std(ddof=1).iloc[-1]
            return float(volatility) if pd.notna(volatility) else None
        except Exception as e: logger.error(f"Volatility calc error: {e}", exc_info=False); return None

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> float | None:
        if close_prices is None or period <= 0 or len(close_prices) < period: return None
        try:
            ema = close_prices.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            return float(ema) if pd.notna(ema) else None
        except Exception as e: logger.error(f"EMA-{period} calc error: {e}", exc_info=False); return None

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> float | None:
        if close_prices is None or period <= 0 or len(close_prices) < period + 1: return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).fillna(0)  # Fill first NaN gain
            loss = -delta.where(delta < 0, 0.0).fillna(0)  # Fill first NaN loss
            avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-12)  # Add epsilon
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_val = rsi.iloc[-1]
            return float(max(0.0, min(100.0, rsi_val))) if pd.notna(rsi_val) else None
        except Exception as e: logger.error(f"RSI-{period} calc error: {e}", exc_info=False); return None

    @staticmethod
    def calculate_macd(close_prices: pd.Series, short_p: int, long_p: int, signal_p: int) -> tuple[float | None, float | None, float | None]:
        min_len = long_p + signal_p  # Rough estimate
        if close_prices is None or not all(p > 0 for p in [short_p, long_p, signal_p]) or short_p >= long_p or len(close_prices) < min_len: return None, None, None
        try:
            ema_short = close_prices.ewm(span=short_p, adjust=False, min_periods=short_p).mean()
            ema_long = close_prices.ewm(span=long_p, adjust=False, min_periods=long_p).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal_p, adjust=False, min_periods=signal_p).mean()
            histogram = macd_line - signal_line
            macd_val, signal_val, hist_val = macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
            if any(pd.isna(v) for v in [macd_val, signal_val, hist_val]): return None, None, None
            return float(macd_val), float(signal_val), float(hist_val)
        except Exception as e: logger.error(f"MACD calc error: {e}", exc_info=False); return None, None, None

    @staticmethod
    def calculate_rsi_series(close_prices: pd.Series, period: int) -> pd.Series | None:
        """Helper to calculate the full RSI series needed for StochRSI."""
        if close_prices is None or period <= 0 or len(close_prices) < period + 1: return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).fillna(0)
            loss = -delta.where(delta < 0, 0.0).fillna(0)
            avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-12)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi.clip(0, 100)  # Clip entire series
        except Exception as e: logger.error(f"RSI Series calc error: {e}", exc_info=False); return None

    @staticmethod
    def calculate_stoch_rsi(close_prices: pd.Series, rsi_p: int, stoch_p: int, k_p: int, d_p: int) -> tuple[float | None, float | None]:
        min_len_est = rsi_p + stoch_p + max(k_p, d_p)  # Rough estimate
        if close_prices is None or not all(p > 0 for p in [rsi_p, stoch_p, k_p, d_p]) or len(close_prices) < min_len_est: return None, None
        try:
            rsi_series = ScalpingBot.calculate_rsi_series(close_prices, rsi_p)
            if rsi_series is None or rsi_series.isna().all(): return None, None
            rsi_series = rsi_series.dropna()
            if len(rsi_series) < stoch_p + max(k_p, d_p): return None, None  # Check length after dropping NaNs

            min_rsi = rsi_series.rolling(window=stoch_p, min_periods=stoch_p).min()
            max_rsi = rsi_series.rolling(window=stoch_p, min_periods=stoch_p).max()
            stoch_rsi_raw = (100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-12)).clip(0, 100)

            stoch_k = stoch_rsi_raw.rolling(window=k_p, min_periods=k_p).mean()
            stoch_d = stoch_k.rolling(window=d_p, min_periods=d_p).mean()
            k_val, d_val = stoch_k.iloc[-1], stoch_d.iloc[-1]
            if pd.isna(k_val) or pd.isna(d_val): return None, None
            return float(max(0.0, min(100.0, k_val))), float(max(0.0, min(100.0, d_val)))
        except Exception as e: logger.error(f"StochRSI calc error: {e}", exc_info=False); return None, None

    @staticmethod
    def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int) -> float | None:
        if high_prices is None or low_prices is None or close_prices is None or period <= 0 or not (len(high_prices) >= period + 1 and len(low_prices) >= period + 1 and len(close_prices) >= period + 1): return None
        try:
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))
            tr_df = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
            tr = tr_df.max(axis=1, skipna=False)
            # Wilder's smoothing (EMA with alpha = 1/period) is common for ATR
            # com = period - 1 gives alpha = 1 / (1 + com) = 1 / period
            atr = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean().iloc[-1]
            # Alternative: Standard EMA: atr = tr.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            return float(atr) if pd.notna(atr) and atr >= 0 else None
        except Exception as e: logger.error(f"ATR-{period} calc error: {e}", exc_info=False); return None

    # --- Trading Logic & Order Management Methods ---

    def calculate_order_size(
        self,
        current_price: float,
        indicators: dict[str, Any],  # Pass calculated indicators
        signal_score: int | None = None  # Pass signal score for optional adjustment
    ) -> float:
        """Calculates the order size in the BASE currency based on available balance,
        percentage risk, market limits, optional volatility adjustment, and
        optional signal strength adjustment. Enhanced V6.2.

        Args:
            current_price: The current market price.
            indicators: Dictionary of pre-calculated indicators (e.g., {'volatility': 0.01}).
            signal_score: The calculated signal score for optional size adjustment.

        Returns:
            The calculated order size in BASE currency, rounded to exchange precision,
            or 0.0 if checks fail or size is below minimums.
        """
        # --- 0. Pre-computation Checks & Setup ---
        func_name = "calculate_order_size"
        if self.market_info is None:
            logger.error(f"[{func_name}] Market info not loaded. Cannot calculate size.")
            return 0.0
        if current_price <= 1e-12:
            logger.error(f"[{func_name}] Current price ({current_price}) is zero or negative. Cannot calculate size.")
            return 0.0
        if not (0 < self.order_size_percentage <= 1):
             logger.error(f"[{func_name}] Invalid order_size_percentage ({self.order_size_percentage}). Must be > 0 and <= 1.")
             return 0.0

        try:
            quote_currency = self.market_info['quote']
            base_currency = self.market_info['base']
            limits = self.market_info.get('limits', {})
            precision = self.market_info.get('precision', {})
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')
            amount_decimals = self.amount_decimals
            amount_tick_size = precision.get('amount')
        except KeyError as e:
             logger.error(f"[{func_name}] Missing critical market info key: {e}. Cannot calculate size.")
             return 0.0
        except Exception as e:
             logger.error(f"[{func_name}] Error accessing market/precision details: {e}. Cannot calculate size.")
             return 0.0

        logger.debug(f"[{func_name}] Starting size calculation for {base_currency}/{quote_currency} at {self.format_price(current_price)}.")

        # --- 1. Get Available Balance ---
        # fetch_balance now returns None on error
        balance = self.fetch_balance(currency_code=quote_currency)
        if balance is None or balance <= 1e-12:
            # fetch_balance logs error/reason, just log insufficient here
            logger.info(f"[{func_name}] Insufficient balance or failed fetch. Balance: {balance}. Cannot place order.")
            return 0.0
        logger.debug(f"[{func_name}] Available balance: {self.format_price(balance)} {quote_currency}")

        # --- 2. Calculate Base Order Size (Quote Value) ---
        target_quote_value = balance * self.order_size_percentage
        logger.debug(f"[{func_name}] Target Quote Value (Balance * %): {self.format_price(target_quote_value)} {quote_currency} (Risk %: {self.order_size_percentage * 100:.2f}%)")

        # --- 3. Apply Adjustments (Volatility & Signal Strength) ---
        adjustment_factor = 1.0

        # 3a. Volatility Adjustment
        if self.volatility_multiplier > 0:
            volatility = indicators.get('volatility')
            if volatility is not None and volatility > 1e-9:
                vol_adj_factor = 1.0 / (1.0 + volatility * self.volatility_multiplier)
                vol_adj_factor = max(0.1, min(2.0, vol_adj_factor))  # Clamp
                adjustment_factor *= vol_adj_factor
                logger.info(f"[{func_name}] Volatility ({volatility:.5f}) factor: {vol_adj_factor:.3f}. Total Factor: {adjustment_factor:.3f}")
            else: logger.debug(f"[{func_name}] Volatility adjustment skipped: Volatility N/A ({volatility}).")
        else: logger.debug(f"[{func_name}] Volatility adjustment disabled (multiplier={self.volatility_multiplier}).")

        # 3b. Signal Strength Adjustment
        if signal_score is not None and (self.strong_signal_adjustment_factor != 1.0 or self.weak_signal_adjustment_factor != 1.0):
             abs_score = abs(signal_score)
             sig_adj_factor = 1.0
             if abs_score >= STRONG_SIGNAL_THRESHOLD_ABS: sig_adj_factor = self.strong_signal_adjustment_factor; adj_type = "Strong"
             # We assume this func is called only *after* score >= ENTRY_SIGNAL_THRESHOLD_ABS check
             # So, weak signal adjustment factor is likely unused here in normal flow.
             elif abs_score >= ENTRY_SIGNAL_THRESHOLD_ABS: sig_adj_factor = 1.0; adj_type = "Normal"  # Normal entry = 1.0x unless configured otherwise
             else: sig_adj_factor = self.weak_signal_adjustment_factor; adj_type = "Weak"  # Should not happen

             if abs(sig_adj_factor - 1.0) > 1e-9:
                 adjustment_factor *= sig_adj_factor
                 logger.info(f"[{func_name}] Signal Score ({signal_score}, {adj_type}) factor: {sig_adj_factor:.3f}. Total Factor: {adjustment_factor:.3f}")
             else: logger.debug(f"[{func_name}] Signal Score ({signal_score}, {adj_type}) factor is {sig_adj_factor:.3f}, no adjustment.")
        else: logger.debug(f"[{func_name}] Signal score adjustment skipped.")

        # Clamp final adjustment factor
        adjustment_factor = max(0.05, min(2.5, adjustment_factor))  # Clamp: 0.05x to 2.5x
        if abs(adjustment_factor - 1.0) > 1e-9: logger.debug(f"[{func_name}] Final clamped adjustment factor: {adjustment_factor:.3f}")

        final_quote_value = target_quote_value * adjustment_factor
        if final_quote_value <= 1e-12:
             logger.warning(f"[{func_name}] Quote value zero/negative after adjustments. Cannot proceed.")
             return 0.0
        logger.debug(f"[{func_name}] Final adjusted quote value: {self.format_price(final_quote_value)} {quote_currency}")

        # --- 4. Convert Quote Value to Base Amount ---
        order_size_base_raw = final_quote_value / current_price
        logger.debug(f"[{func_name}] Raw base amount: {self.format_amount(order_size_base_raw, amount_decimals + 4)} {base_currency}")

        # --- 5. Apply Exchange Precision AND Check Limits ---
        try:
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base_raw)
            amount_precise = float(amount_precise_str)
            logger.debug(f"[{func_name}] Amount after precision ({amount_tick_size}): {self.format_amount(amount_precise)} {base_currency}")

            if amount_precise <= 1e-12:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Amount zero after precision ({amount_tick_size}). Raw: {self.format_amount(order_size_base_raw, amount_decimals + 4)}. Cannot place order.{Style.RESET_ALL}")
                 return 0.0
            if min_amount is not None and amount_precise < min_amount:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Precise amount {self.format_amount(amount_precise)} {base_currency} < Min Amount {self.format_amount(min_amount)}. Cannot place order.{Style.RESET_ALL}")
                return 0.0

            price_precise_str = self.exchange.price_to_precision(self.symbol, current_price)
            price_precise = float(price_precise_str)
            estimated_cost = amount_precise * price_precise

            if min_cost is not None and estimated_cost < min_cost:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Estimated cost {self.format_price(estimated_cost)} {quote_currency} < Min Cost {self.format_price(min_cost)}. Cannot place order.{Style.RESET_ALL}")
                return 0.0

            # --- Success ---
            logger.info(f"{Fore.CYAN}[{func_name}] Calculated final order size: {self.format_amount(amount_precise)} {base_currency} (Est. Cost: {self.format_price(estimated_cost)} {quote_currency}){Style.RESET_ALL}")
            return amount_precise

        except (ccxt.InvalidOrder, ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Error applying precision/limits: {type(e).__name__} - {e}. Raw: {self.format_amount(order_size_base_raw, amount_decimals + 4)}.{Style.RESET_ALL}")
            return 0.0
        except Exception as e:
             logger.error(f"{Fore.RED}[{func_name}] Unexpected error in size precision/limit checks: {e}{Style.RESET_ALL}", exc_info=True)
             return 0.0

    def _calculate_sl_tp_prices(self, entry_price: float, side: str, current_price: float, atr: float | None) -> tuple[float | None, float | None]:
        """Calculates SL/TP prices based on config, applies precision and sanity checks."""
        func_name = "_calculate_sl_tp_prices"
        stop_loss_price_raw: float | None = None
        take_profit_price_raw: float | None = None

        if self.market_info is None or entry_price <= 1e-12:
            logger.error(f"[{func_name}] Market info missing or invalid entry price ({entry_price}).")
            return None, None

        # --- Calculate Raw Prices ---
        if self.use_atr_sl_tp:
            if atr is None or atr <= 1e-12:
                logger.warning(f"{Fore.YELLOW}[{func_name}] ATR SL/TP enabled, but ATR invalid ({atr}). Cannot calculate.{Style.RESET_ALL}")
                return None, None
            if not (self.atr_sl_multiplier > 0 and self.atr_tp_multiplier > 0):
                 logger.warning(f"{Fore.YELLOW}[{func_name}] ATR SL/TP enabled, but multipliers invalid (SL={self.atr_sl_multiplier}, TP={self.atr_tp_multiplier}). Cannot calculate.{Style.RESET_ALL}")
                 return None, None
            logger.debug(f"[{func_name}] Calculating SL/TP using ATR={self.format_price(atr)}, SL Mult={self.atr_sl_multiplier}, TP Mult={self.atr_tp_multiplier}")
            sl_delta, tp_delta = atr * self.atr_sl_multiplier, atr * self.atr_tp_multiplier
            stop_loss_price_raw = entry_price - sl_delta if side == "buy" else entry_price + sl_delta
            take_profit_price_raw = entry_price + tp_delta if side == "buy" else entry_price - tp_delta
        else:  # Fixed Percentage
            sl_pct, tp_pct = self.base_stop_loss_pct, self.base_take_profit_pct
            if not (sl_pct and tp_pct and 0 < sl_pct < 1 and 0 < tp_pct < 1):
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Fixed % SL/TP enabled, but percentages invalid (SL={sl_pct}, TP={tp_pct}). Cannot calculate.{Style.RESET_ALL}")
                 return None, None
            logger.debug(f"[{func_name}] Calculating SL/TP using Fixed %: SL={sl_pct * 100:.2f}%, TP={tp_pct * 100:.2f}%")
            stop_loss_price_raw = entry_price * (1 - sl_pct) if side == "buy" else entry_price * (1 + sl_pct)
            take_profit_price_raw = entry_price * (1 + tp_pct) if side == "buy" else entry_price * (1 - tp_pct)

        # --- Apply Precision and Sanity Checks ---
        stop_loss_price_final: float | None = None
        take_profit_price_final: float | None = None
        try:
            # Process SL
            if stop_loss_price_raw is not None and stop_loss_price_raw > 1e-12:
                 sl_precise = float(self.exchange.price_to_precision(self.symbol, stop_loss_price_raw))
                 if (side == 'buy' and sl_precise < entry_price) or \
                    (side == 'sell' and sl_precise > entry_price):
                      stop_loss_price_final = sl_precise
                 else: logger.warning(f"{Fore.YELLOW}[{func_name}] SL {self.format_price(sl_precise)} invalid vs entry {self.format_price(entry_price)}. SL set to None.{Style.RESET_ALL}")
            elif stop_loss_price_raw is not None: logger.warning(f"[{func_name}] Raw SL price ({stop_loss_price_raw}) zero/negative. SL set to None.")

            # Process TP
            if take_profit_price_raw is not None and take_profit_price_raw > 1e-12:
                 tp_precise = float(self.exchange.price_to_precision(self.symbol, take_profit_price_raw))
                 if (side == 'buy' and tp_precise > entry_price) or \
                    (side == 'sell' and tp_precise < entry_price):
                      take_profit_price_final = tp_precise
                 else: logger.warning(f"{Fore.YELLOW}[{func_name}] TP {self.format_price(tp_precise)} invalid vs entry {self.format_price(entry_price)}. TP set to None.{Style.RESET_ALL}")
            elif take_profit_price_raw is not None: logger.warning(f"[{func_name}] Raw TP price ({take_profit_price_raw}) zero/negative. TP set to None.")

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Error applying precision to SL/TP: {e}. Raw SL={stop_loss_price_raw}, TP={take_profit_price_raw}. Setting both None.{Style.RESET_ALL}")
            return None, None
        except Exception as e:
             logger.error(f"{Fore.RED}[{func_name}] Unexpected error in SL/TP precision: {e}. Both set to None.{Style.RESET_ALL}", exc_info=True)
             return None, None

        # Final Check: SL vs TP
        if stop_loss_price_final is not None and take_profit_price_final is not None:
             if (side == "buy" and stop_loss_price_final >= take_profit_price_final) or \
                (side == "sell" and stop_loss_price_final <= take_profit_price_final):
                  logger.warning(f"{Fore.YELLOW}[{func_name}] Final SL {self.format_price(stop_loss_price_final)} conflicts with TP {self.format_price(take_profit_price_final)}. Setting TP to None.{Style.RESET_ALL}")
                  take_profit_price_final = None

        logger.debug(f"[{func_name}] Final calculated SL={self.format_price(stop_loss_price_final)}, TP={self.format_price(take_profit_price_final)}")
        return stop_loss_price_final, take_profit_price_final

    def compute_trade_signal_score(self, price: float, indicators: dict[str, Any], orderbook_imbalance: float | None) -> tuple[int, list[str]]:
        """Computes trade signal score based on indicators and order book imbalance."""
        func_name = "compute_trade_signal_score"
        score = 0.0
        reasons = []
        RSI_OS, RSI_OB = 35, 65
        STOCH_OS, STOCH_OB = 25, 75
        EMA_THRESH_MULT = 0.0002  # Price 0.02% away from EMA

        # 1. Order Book Imbalance
        imb_str = "N/A"
        if orderbook_imbalance is not None:
            imb = orderbook_imbalance
            imb_str = f"{imb:.2f}" if imb != float('inf') else "Inf"
            if self.imbalance_threshold <= 0: reason = "[ 0.0] OB Invalid Threshold"
            elif imb == float('inf'): score -= 1.0; reason = f"{Fore.RED}[-1.0] OB Sell (Imb: Inf){Style.RESET_ALL}"
            else:
                 imb_buy_thresh = 1.0 / self.imbalance_threshold
                 if imb < imb_buy_thresh: score += 1.0; reason = f"{Fore.GREEN}[+1.0] OB Buy (Imb < {imb_buy_thresh:.2f}){Style.RESET_ALL}"
                 elif imb > self.imbalance_threshold: score -= 1.0; reason = f"{Fore.RED}[-1.0] OB Sell (Imb > {self.imbalance_threshold:.2f}){Style.RESET_ALL}"
                 else: reason = f"{Fore.WHITE}[ 0.0] OB Neutral{Style.RESET_ALL}"
        else: reason = f"{Fore.WHITE}[ 0.0] OB N/A{Style.RESET_ALL}"
        reasons.append(f"{reason} (Val: {imb_str})")

        # 2. EMA Trend
        ema = indicators.get('ema')
        ema_str = self.format_price(ema)
        if ema is not None and ema > 1e-9:
            if price > ema * (1 + EMA_THRESH_MULT): score += 1.0; reason = f"{Fore.GREEN}[+1.0] Price > EMA{Style.RESET_ALL}"
            elif price < ema * (1 - EMA_THRESH_MULT): score -= 1.0; reason = f"{Fore.RED}[-1.0] Price < EMA{Style.RESET_ALL}"
            else: reason = f"{Fore.WHITE}[ 0.0] Price ~ EMA{Style.RESET_ALL}"
        else: reason = f"{Fore.WHITE}[ 0.0] EMA N/A{Style.RESET_ALL}"
        reasons.append(f"{reason} (EMA: {ema_str}, Price: {self.format_price(price)})")

        # 3. RSI Momentum/OB/OS
        rsi = indicators.get('rsi')
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        if rsi is not None:
            if rsi < RSI_OS: score += 1.0; reason = f"{Fore.GREEN}[+1.0] RSI Oversold (<{RSI_OS}){Style.RESET_ALL}"
            elif rsi > RSI_OB: score -= 1.0; reason = f"{Fore.RED}[-1.0] RSI Overbought (>{RSI_OB}){Style.RESET_ALL}"
            else: reason = f"{Fore.WHITE}[ 0.0] RSI Neutral{Style.RESET_ALL}"
        else: reason = f"{Fore.WHITE}[ 0.0] RSI N/A{Style.RESET_ALL}"
        reasons.append(f"{reason} (Val: {rsi_str})")

        # 4. MACD Momentum/Cross
        macd_line, macd_signal = indicators.get('macd_line'), indicators.get('macd_signal')
        macd_str = f"L:{self.format_price(macd_line, self.price_decimals + 1)}/S:{self.format_price(macd_signal, self.price_decimals + 1)}" if macd_line is not None else "N/A"
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal: score += 1.0; reason = f"{Fore.GREEN}[+1.0] MACD Line > Signal{Style.RESET_ALL}"
            else: score -= 1.0; reason = f"{Fore.RED}[-1.0] MACD Line <= Signal{Style.RESET_ALL}"
        else: reason = f"{Fore.WHITE}[ 0.0] MACD N/A{Style.RESET_ALL}"
        reasons.append(f"{reason} ({macd_str})")

        # 5. Stochastic RSI OB/OS
        stoch_k, stoch_d = indicators.get('stoch_k'), indicators.get('stoch_d')
        stoch_str = f"K:{stoch_k:.1f}/D:{stoch_d:.1f}" if stoch_k is not None else "N/A"
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < STOCH_OS and stoch_d < STOCH_OS: score += 1.0; reason = f"{Fore.GREEN}[+1.0] StochRSI Oversold (<{STOCH_OS}){Style.RESET_ALL}"
            elif stoch_k > STOCH_OB and stoch_d > STOCH_OB: score -= 1.0; reason = f"{Fore.RED}[-1.0] StochRSI Overbought (>{STOCH_OB}){Style.RESET_ALL}"
            else: reason = f"{Fore.WHITE}[ 0.0] StochRSI Neutral{Style.RESET_ALL}"
        else: reason = f"{Fore.WHITE}[ 0.0] StochRSI N/A{Style.RESET_ALL}"
        reasons.append(f"{reason} ({stoch_str})")

        # --- Final Score ---
        final_score = int(round(score))
        logger.debug(f"[{func_name}] Raw Score: {score:.2f}, Final Integer Score: {final_score}")
        return final_score, reasons

    @retry_api_call(max_retries=2, initial_delay=2)
    def place_entry_order(
        self, side: str, order_size_base: float, confidence_level: int,
        order_type: str, current_price: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None
    ) -> dict[str, Any] | None:
        """Places entry order (market/limit) with Bybit V5 SL/TP params."""
        func_name = "place_entry_order"
        if self.market_info is None: logger.error(f"[{func_name}] Market info missing."); return None
        if order_size_base <= 0: logger.error(f"[{func_name}] Invalid order size {order_size_base}."); return None
        if current_price <= 1e-12: logger.error(f"[{func_name}] Invalid current price {current_price}."); return None

        base_currency = self.market_info['base']
        quote_currency = self.market_info['quote']
        params = {}
        limit_price: float | None = None

        try:
            amount_precise = order_size_base  # Already validated & precise
            logger.debug(f"[{func_name}] Using validated precise amount: {self.format_amount(amount_precise)} {base_currency}")

            if order_type == "limit":
                offset = self.limit_order_entry_offset_pct_buy if side == 'buy' else self.limit_order_entry_offset_pct_sell
                price_factor = (1 - offset) if side == 'buy' else (1 + offset)
                limit_price_raw = current_price * price_factor
                if limit_price_raw <= 1e-12: raise ValueError("Calculated limit price zero/negative")
                limit_price = float(self.exchange.price_to_precision(self.symbol, limit_price_raw))
                logger.debug(f"[{func_name}] Calculated precise limit price: {self.format_price(limit_price)}")
                if (side == 'buy' and limit_price >= current_price) or (side == 'sell' and limit_price <= current_price):
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Limit price {self.format_price(limit_price)} not favorable vs current {self.format_price(current_price)} for {side}. Check offset.{Style.RESET_ALL}")

            # Add Bybit V5 SL/TP Params (as strings)
            if stop_loss_price is not None:
                params['stopLoss'] = self.format_price(stop_loss_price)  # Format to string with correct decimals
                if self.sl_trigger_by: params['slTriggerBy'] = self.sl_trigger_by
            if take_profit_price is not None:
                params['takeProfit'] = self.format_price(take_profit_price)
                if self.tp_trigger_by: params['tpTriggerBy'] = self.tp_trigger_by
            if params.get('stopLoss') or params.get('takeProfit'):
                 logger.debug(f"[{func_name}] Adding SL/TP params: {params}")

            # Add Bybit V5 category param
            if self.exchange_id.lower() == 'bybit':
                market_type = self.market_info.get('type', 'swap')
                is_linear = self.market_info.get('linear', True)
                if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                elif market_type == 'spot': params['category'] = 'spot'
                if 'category' in params: logger.debug(f"[{func_name}] Added Bybit category: {params['category']}")

        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Error preparing order values/params: {e}{Style.RESET_ALL}", exc_info=True)
            return None
        except Exception as e:
             logger.error(f"{Fore.RED}[{func_name}] Unexpected error preparing order: {e}{Style.RESET_ALL}", exc_info=True)
             return None

        # --- Log Order Details Before Placing ---
        log_color = Fore.GREEN if side == 'buy' else Fore.RED
        action_desc = f"{order_type.upper()} {side.upper()} ENTRY"
        sl_info = f"SL={params.get('stopLoss', 'N/A')}" + (f" ({params['slTriggerBy']})" if 'slTriggerBy' in params else "")
        tp_info = f"TP={params.get('takeProfit', 'N/A')}" + (f" ({params['tpTriggerBy']})" if 'tpTriggerBy' in params else "")
        limit_info = f"Limit={self.format_price(limit_price)}" if limit_price else f"Market (~{self.format_price(current_price)})"
        est_value = amount_precise * (limit_price or current_price)

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_id = f"sim_entry_{int(time.time() * 1000)}_{side[:1]}"
            sim_entry_price = limit_price if order_type == "limit" else current_price
            sim_status = 'open' if order_type == 'limit' else 'closed'  # Market assumed filled
            sim_filled = amount_precise if sim_status == 'closed' else 0.0
            sim_avg = sim_entry_price if sim_status == 'closed' else None
            sim_ts = int(time.time() * 1000)

            simulated_order = {
                "id": sim_id, "timestamp": sim_ts, "datetime": pd.to_datetime(sim_ts, unit='ms', utc=True).isoformat(),
                "symbol": self.symbol, "type": order_type, "side": side, "status": sim_status,
                "price": limit_price, "amount": amount_precise,
                "filled": sim_filled, "remaining": amount_precise - sim_filled,
                "average": sim_avg, "cost": sim_filled * sim_avg if sim_avg else 0.0,
                "stopLossPrice": stop_loss_price, "takeProfitPrice": take_profit_price,
                "info": {"simulated": True, "orderId": sim_id, **params},  # Include params sent
                "bot_custom_info": {"confidence": confidence_level, "initial_base_size": order_size_base}
            }
            logger.info(
                f"{log_color}[SIMULATION] Placing {action_desc}: "
                f"ID: {sim_id}, Size: {self.format_amount(amount_precise)} {base_currency}, Price: {limit_info}, "
                f"EstVal: {self.format_price(est_value)} {quote_currency}, Conf: {confidence_level}, {sl_info}, {tp_info}{Style.RESET_ALL}"
            )
            return simulated_order

        # --- Live Trading Mode ---
        else:
            logger.info(f"{log_color}Attempting to place LIVE {action_desc} order...")
            log_details = (f" -> Size: {self.format_amount(amount_precise)} {base_currency}\n"
                           f" -> Price: {limit_info}\n"
                           f" -> Value: ~{self.format_price(est_value)} {quote_currency}\n"
                           f" -> Confidence: {confidence_level}\n"
                           f" -> Params: {params}")
            logger.info(log_details)

            order: dict[str, Any] | None = None
            try:
                if order_type == "market":
                    order = self.exchange.create_market_order(self.symbol, side, amount_precise, params=params)
                elif order_type == "limit":
                    if limit_price is None: raise ValueError("Limit price is None for limit order")
                    order = self.exchange.create_limit_order(self.symbol, side, amount_precise, limit_price, params=params)
                else: raise ValueError(f"Unsupported live order type '{order_type}'")

                if order:
                    oid = order.get('id', 'N/A')
                    ostatus = order.get('status', STATUS_UNKNOWN)
                    ofilled = order.get('filled', 0.0)
                    oavg = order.get('average')
                    info_sl = order.get('info', {}).get('stopLoss', 'N/A')  # Check confirmation
                    info_tp = order.get('info', {}).get('takeProfit', 'N/A')

                    logger.info(
                        f"{log_color}---> LIVE {action_desc} Order Placed: ID: {oid}, Status: {ostatus}, "
                        f"Filled: {self.format_amount(ofilled)}, AvgPrice: {self.format_price(oavg)}, "
                        f"SL Sent/Conf: {params.get('stopLoss', 'N/A')}/{info_sl}, "
                        f"TP Sent/Conf: {params.get('takeProfit', 'N/A')}/{info_tp}{Style.RESET_ALL}"
                    )
                    order['bot_custom_info'] = {"confidence": confidence_level, "initial_base_size": order_size_base}
                    return order
                else:
                    # Should be caught by retry decorator returning None, but handle defensively
                    logger.error(f"{Fore.RED}LIVE {action_desc} order placement API call returned None. Check exchange status/logs.{Style.RESET_ALL}")
                    return None

            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                logger.error(f"{Fore.RED}LIVE {action_desc} Failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
                if isinstance(e, ccxt.InsufficientFunds): self.fetch_balance(quote_currency)  # Log current balance
                return None
            except ccxt.ExchangeError as e:
                 logger.error(f"{Fore.RED}LIVE {action_desc} Failed (ExchangeError): {e}{Style.RESET_ALL}")
                 return None
            except Exception as e:
                logger.error(f"{Fore.RED}LIVE {action_desc} Failed (Unexpected Python Error): {e}{Style.RESET_ALL}", exc_info=True)
                return None

    @retry_api_call(max_retries=1)
    def cancel_order_by_id(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancels order by ID. Returns True if cancelled or already gone, False on failure."""
        func_name = "cancel_order_by_id"
        if not order_id: logger.warning(f"[{func_name}] Empty order_id."); return False

        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}Attempting to cancel order {order_id} ({target_symbol})...{Style.RESET_ALL}")

        if self.simulation_mode:
            logger.info(f"[SIMULATION][{func_name}] Simulating cancel for {order_id}.")
            # Update state directly in caller if needed after simulation success
            return True  # Assume simulation always works

        else:  # Live Mode
            try:
                params = {}
                if self.exchange_id.lower() == 'bybit':
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    is_linear = self.exchange.market(target_symbol).get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'

                response = self.exchange.cancel_order(order_id, target_symbol, params=params)
                logger.info(f"{Fore.YELLOW}---> Cancellation initiated for order {order_id}. Response: {str(response)[:100]}...{Style.RESET_ALL}")
                return True
            except ccxt.OrderNotFound:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} not found (already closed/cancelled?). Treating as success.{Style.RESET_ALL}")
                return True
            except ccxt.NetworkError as e: raise e  # Let retry handle
            except ccxt.ExchangeError as e:
                 err_str = str(e).lower()
                 if any(p in err_str for p in ["order has been filled", "order is finished", "already closed", "already cancelled", "cannot be cancelled", "order status error"]):
                      logger.warning(f"{Fore.YELLOW}[{func_name}] Cannot cancel {order_id}: Already final state. Error: {e}{Style.RESET_ALL}")
                      return True  # Treat as 'success' (no action needed)
                 else:
                      logger.error(f"{Fore.RED}[{func_name}] Exchange error cancelling {order_id}: {e}{Style.RESET_ALL}")
                      return False
            except Exception as e:
                logger.error(f"{Fore.RED}[{func_name}] Unexpected error cancelling {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
                return False

    @retry_api_call()
    def cancel_all_symbol_orders(self, symbol: str | None = None) -> int:
        """Cancels all open orders for symbol. Returns count or -1."""
        func_name = "cancel_all_symbol_orders"
        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}Attempting to cancel all OPEN orders for {target_symbol}...{Style.RESET_ALL}")
        cancelled_count = 0

        if self.simulation_mode:
             logger.info(f"[SIMULATION][{func_name}] Simulating cancel all for {target_symbol}.")
             # State update should happen in caller based on this simulated success
             return 0  # Indicate simulated action taken, count handled in caller

        else:  # Live Mode
            try:
                params = {}
                if self.exchange_id.lower() == 'bybit':
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    is_linear = self.exchange.market(target_symbol).get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'
                    # Optional filter for Bybit (check docs):
                    # params['orderFilter'] = 'Order' # To cancel only regular orders?
                    # params['stopOrderType'] = 'Stop' # Or 'StopLoss', 'TakeProfit'?

                # Prefer unified cancelAllOrders if available
                if self.exchange.has['cancelAllOrders']:
                    logger.debug(f"[{func_name}] Using unified 'cancelAllOrders' for {target_symbol}...")
                    response = self.exchange.cancel_all_orders(target_symbol, params=params)
                    logger.info(f"[{func_name}] 'cancelAllOrders' response: {str(response)[:100]}...")
                    # Count is often unknown from this method
                    cancelled_count = -1  # Indicate unknown count, action attempted
                # Fallback to fetch + individual cancel (more reliable feedback)
                elif self.exchange.has['fetchOpenOrders']:
                    logger.warning(f"[{func_name}] 'cancelAllOrders' unavailable, using fetchOpenOrders + individual cancel for {target_symbol}...")
                    open_orders = self.exchange.fetch_open_orders(target_symbol, params=params)  # Use same params
                    if not open_orders:
                        logger.info(f"[{func_name}] No open orders found via fetchOpenOrders for {target_symbol}.")
                        return 0

                    logger.warning(f"{Fore.YELLOW}[{func_name}] Found {len(open_orders)} open order(s). Cancelling individually...{Style.RESET_ALL}")
                    for order in open_orders:
                        if self.cancel_order_by_id(order['id'], target_symbol):
                            cancelled_count += 1
                            time.sleep(max(0.2, self.exchange.rateLimit / 1000))  # Rate limit delay
                        else: logger.error(f"[{func_name}] Failed to cancel order {order['id']} during bulk attempt.")
                else:
                     logger.error(f"{Fore.RED}[{func_name}] Exchange lacks 'cancelAllOrders' and 'fetchOpenOrders'. Cannot cancel automatically.{Style.RESET_ALL}")
                     return 0

                final_msg = f"[{func_name}] Bulk cancel process finished for {target_symbol}. "
                final_msg += f"Result: {cancelled_count}" + (" (unknown count)" if cancelled_count == -1 else " orders")
                logger.info(final_msg)
                return cancelled_count

            except Exception as e:
                logger.error(f"{Fore.RED}[{func_name}] Error during bulk cancel for {target_symbol}: {e}{Style.RESET_ALL}", exc_info=True)
                return cancelled_count  # Return count successful before error

    # --- Position Management & State Update Logic ---

    def _check_pending_entries(self, indicators: dict) -> None:
        """Checks status of pending limit orders, updates state if filled."""
        func_name = "_check_pending_entries"
        pending_positions = [pos for pos in self.open_positions if pos.get('status') == STATUS_PENDING_ENTRY]
        if not pending_positions: return

        logger.debug(f"[{func_name}] Checking status of {len(pending_positions)} pending entry order(s)...")
        current_price_for_check: float | None = None
        positions_to_remove_ids = set()
        positions_to_update_data = {}  # {order_id: updated_position_dict}

        for position in pending_positions:
            entry_order_id = position.get('id')
            pos_symbol = position.get('symbol', self.symbol)
            if not entry_order_id: logger.error(f"[{func_name}] Pending position missing ID: {position}"); continue

            order_info = self.fetch_order_status(entry_order_id, symbol=pos_symbol)

            # Case 1: Order Not Found / Vanished
            if order_info is None:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Pending order {entry_order_id} not found. Assuming closed/cancelled externally. Removing.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)
                continue

            order_status = order_info.get('status')
            filled_amount = float(order_info.get('filled', 0.0))
            entry_price = order_info.get('average')  # Average fill price

            # Case 2: Order Fully Filled ('closed' status)
            if order_status == 'closed' and filled_amount > 1e-12:
                if entry_price is None or float(entry_price) <= 1e-12:
                    logger.error(f"{Fore.RED}[{func_name}] Order {entry_order_id} 'closed' but invalid fill price ({entry_price}). Cannot activate. Removing.{Style.RESET_ALL}")
                    positions_to_remove_ids.add(entry_order_id)
                    continue

                entry_price = float(entry_price)
                orig_size = position.get('original_size')
                if orig_size is not None and abs(filled_amount - orig_size) > max(orig_size * 0.01, 1e-9):
                     logger.warning(f"{Fore.YELLOW}[{func_name}] Filled amount {self.format_amount(filled_amount)} differs from requested {self.format_amount(orig_size)}. Using actual.{Style.RESET_ALL}")

                logger.info(f"{Fore.GREEN}---> Pending entry order {entry_order_id} FILLED! Amount: {self.format_amount(filled_amount)}, Avg Price: {self.format_price(entry_price)}{Style.RESET_ALL}")

                updated_pos = position.copy()
                updated_pos['size'] = filled_amount  # Use actual filled amount
                updated_pos['entry_price'] = entry_price
                updated_pos['status'] = STATUS_ACTIVE
                fill_time_ms = order_info.get('lastTradeTimestamp') or order_info.get('timestamp')
                updated_pos['entry_time'] = fill_time_ms / 1000 if fill_time_ms else time.time()
                updated_pos['last_update_time'] = time.time()

                # Recalculate internal SL/TP based on actual fill price
                if current_price_for_check is None: current_price_for_check = self.fetch_market_price()
                if current_price_for_check:
                    atr_value = indicators.get('atr')
                    sl_price, tp_price = self._calculate_sl_tp_prices(
                        entry_price=updated_pos['entry_price'], side=updated_pos['side'],
                        current_price=current_price_for_check, atr=atr_value
                    )
                    updated_pos['stop_loss_price'] = sl_price
                    updated_pos['take_profit_price'] = tp_price
                    logger.info(f"[{func_name}] Stored internal SL={self.format_price(sl_price)}, TP={self.format_price(tp_price)} for activated pos {entry_order_id}.")
                else: logger.warning(f"[{func_name}] Could not fetch price for SL/TP recalc after fill for {entry_order_id}.")

                positions_to_update_data[entry_order_id] = updated_pos

            # Case 3: Order Failed (canceled, rejected, expired)
            elif order_status in ['canceled', 'rejected', 'expired']:
                reason = order_info.get('info', {}).get('rejectReason', order_info.get('info', {}).get('cancelType', order_status))
                logger.warning(f"{Fore.YELLOW}[{func_name}] Pending order {entry_order_id} failed (Status: {order_status}, Reason: {reason}). Removing.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)

            # Case 4: Still Open or Partially Filled
            elif order_status == 'open':
                 logger.debug(f"[{func_name}] Pending order {entry_order_id} still 'open'. Filled: {self.format_amount(filled_amount)}. Waiting...")
                 # Update timestamp maybe?
                 # positions_to_update_data[entry_order_id] = {'last_update_time': time.time()} # Only update timestamp

            # Case 5: Closed but zero filled (treat as cancelled/failed)
            elif order_status == 'closed' and filled_amount <= 1e-12:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Pending order {entry_order_id} 'closed' but zero filled. Assuming cancelled pre-fill. Removing.{Style.RESET_ALL}")
                 positions_to_remove_ids.add(entry_order_id)

            else:  # Unexpected status
                 logger.warning(f"[{func_name}] Pending order {entry_order_id} has unexpected status: {order_status}. Filled: {self.format_amount(filled_amount)}. Leaving pending.")

        # --- Apply State Updates Atomically ---
        if positions_to_remove_ids or positions_to_update_data:
            new_positions_list = []
            removed_count = 0
            updated_count = 0
            for pos in self.open_positions:
                pos_id = pos.get('id')
                if pos_id in positions_to_remove_ids:
                    removed_count += 1; continue
                if pos_id in positions_to_update_data:
                    # Use updated dict for filled orders, or merge simple updates like timestamp
                    if pos['status'] == STATUS_PENDING_ENTRY and positions_to_update_data[pos_id]['status'] == STATUS_ACTIVE:
                         new_positions_list.append(positions_to_update_data[pos_id])  # Full replace
                         updated_count += 1
                    else:
                         pos.update(positions_to_update_data[pos_id])  # Partial update (e.g., timestamp)
                         new_positions_list.append(pos)
                else:
                    new_positions_list.append(pos)  # Keep unchanged

            if removed_count > 0 or updated_count > 0:
                self.open_positions = new_positions_list
                logger.debug(f"[{func_name}] Checks complete. Activated: {updated_count}, Removed: {removed_count}. Total positions: {len(self.open_positions)}")
                self._save_state()

    def _manage_active_positions(self, current_price: float, indicators: dict) -> None:
        """Manages active positions: Checks closure, Time Exit, TSL (Experimental)."""
        func_name = "_manage_active_positions"
        active_positions = [pos for pos in self.open_positions if pos.get('status') == STATUS_ACTIVE]
        if not active_positions: return

        logger.debug(f"[{func_name}] Managing {len(active_positions)} active position(s) against price: {self.format_price(current_price)}...")
        positions_to_remove_ids = set()  # IDs of positions confirmed closed
        positions_to_update_state = {}  # {pos_id: {key: value}} for TSL updates etc.

        for position in active_positions:
            pos_id = position.get('id')
            if not pos_id: logger.error(f"[{func_name}] Active position missing ID: {position}"); continue

            # --- Get Position Details ---
            symbol = position.get('symbol', self.symbol)
            side = position.get('side')
            entry_price = position.get('entry_price')
            position_size = position.get('size')
            entry_time = position.get('entry_time')

            # Basic validation
            if not all([symbol, side, entry_price, position_size, entry_time]):
                 logger.error(f"[{func_name}] Active pos {pos_id} missing essential data. Skipping."); continue
            if position_size <= 1e-12:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Active pos {pos_id} size ~0. Assuming closed. Removing.{Style.RESET_ALL}")
                 positions_to_remove_ids.add(pos_id); continue

            exit_reason: str | None = None
            exit_price: float | None = None

            # --- 1. Check for External Closure (SL/TP/Manual) ---
            # Primary Check: fetch_order_status of the *entry* order.
            # This is imperfect for parameter-based SL/TP if they don't affect the original order status.
            # Alternative: Use `fetch_positions` if available and reliable for the exchange.
            # If using `fetch_positions`, need to match based on symbol/side and handle potential discrepancies.
            # Sticking with fetch_order_status for now, but be aware of limitations.
            order_info = self.fetch_order_status(pos_id, symbol=symbol)

            if order_info is None:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Active pos main order {pos_id} not found. Assuming closed externally. Removing.{Style.RESET_ALL}")
                exit_reason = f"Main Order Vanished ({pos_id})"
                exit_price = current_price  # Best guess
            elif order_info.get('status') != 'open':
                # If order isn't 'open', it MIGHT mean closure (closed, canceled, etc.)
                # Need to be careful not to misinterpret the initial fill confirmation ('closed') as a closure event.
                order_status = order_info.get('status')
                float(order_info.get('filled', 0.0))
                order_remaining = float(order_info.get('remaining', 0.0))

                # If status is closed/canceled AND remaining amount is zero or very small (close to zero), infer closure.
                # If filled amount also matches position size, it's ambiguous (could be entry confirmation).
                # Heuristic: If status is not 'open' and remaining is zero, assume closed.
                if order_remaining <= 1e-9:  # Check if order is fully settled (no remaining amount)
                    logger.info(f"[{func_name}] Order {pos_id} status '{order_status}' with near-zero remaining. Inferring external closure.")
                    # Try to get a more precise exit price if possible (e.g., from 'average' if status is 'closed')
                    exit_price_raw = order_info.get('average') or order_info.get('price') or current_price
                    try: exit_price = float(exit_price_raw)
                    except (TypeError, ValueError): exit_price = current_price

                    exit_reason = self._infer_exit_reason(order_info, position, exit_price, current_price)
                    log_color = Fore.RED if "SL" in exit_reason else (Fore.GREEN if "TP" in exit_reason else Fore.YELLOW)
                    logger.info(f"{log_color}{exit_reason} at ~{self.format_price(exit_price)}{Style.RESET_ALL}")
                else:
                    # Order status not 'open' but has remaining amount? Unusual state.
                    logger.warning(f"[{func_name}] Order {pos_id} has status '{order_status}' but non-zero remaining ({order_remaining}). State unclear, skipping exit check based on this.")

            # --- If no external closure detected, check bot-managed exits ---
            if not exit_reason:

                # --- 2. Time-Based Exit Check ---
                if self.time_based_exit_minutes is not None and self.time_based_exit_minutes > 0:
                    time_elapsed_min = (time.time() - entry_time) / 60
                    if time_elapsed_min >= self.time_based_exit_minutes:
                        logger.info(f"{Fore.YELLOW}[{func_name}] Time limit ({self.time_based_exit_minutes} min) reached for {pos_id} (Age: {time_elapsed_min:.1f}m). Initiating market close.{Style.RESET_ALL}")
                        market_close_order = self._place_market_close_order(position, current_price)
                        if market_close_order:
                            exit_reason = f"Time Limit ({self.time_based_exit_minutes}min)"
                            exit_price = market_close_order.get('average') or current_price  # Use fill price if available
                            logger.info(f"[{func_name}] Market close for time exit placed for {pos_id}. Exit ~{self.format_price(exit_price)}")
                        else:
                            logger.critical(f"{Fore.RED}{Style.BRIGHT}[{func_name}] CRITICAL FAILURE closing timed-out position {pos_id}. REMAINS OPEN! Manual action required!{Style.RESET_ALL}")
                            # Do NOT set exit_reason, let it retry or require manual action

                # --- 3. Trailing Stop Loss (TSL) Logic [EXPERIMENTAL] ---
                if not exit_reason and self.enable_trailing_stop_loss:
                    if not self.trailing_stop_loss_percentage or not (0 < self.trailing_stop_loss_percentage < 1):
                         logger.error(f"[{func_name}] TSL enabled but percentage invalid. Disabling TSL this cycle.")
                    else:
                        new_tsl_price = self._update_trailing_stop_price(position, current_price)
                        if new_tsl_price is not None:
                            logger.warning(f"{Fore.YELLOW}[EXPERIMENTAL] TSL update for {pos_id}. Attempting edit_order to SL={self.format_price(new_tsl_price)}. VERIFY SUPPORT!{Style.RESET_ALL}")
                            edit_success = self._attempt_edit_order_for_tsl(position, new_tsl_price)
                            if edit_success:
                                update_payload = {
                                    'trailing_stop_price': new_tsl_price,
                                    'stop_loss_price': new_tsl_price,  # Main SL follows TSL
                                    'last_update_time': time.time()
                                }
                                # Merge updates carefully
                                current_updates = positions_to_update_state.get(pos_id, {})
                                positions_to_update_state[pos_id] = {**current_updates, **update_payload}
                                logger.info(f"{Fore.MAGENTA}Internal TSL state for {pos_id} updated to {self.format_price(new_tsl_price)} after edit attempt.{Style.RESET_ALL}")
                            else:
                                 logger.error(f"{Fore.RED}TSL update via edit_order for {pos_id} failed/skipped. TSL state NOT updated.{Style.RESET_ALL}")

            # --- 4. Process Exit and Log PnL ---
            if exit_reason:
                positions_to_remove_ids.add(pos_id)
                self._log_position_pnl(position, exit_price, exit_reason)
                # Mark status in update dict for saving state (though it will be removed)
                current_updates = positions_to_update_state.get(pos_id, {})
                positions_to_update_state[pos_id] = {**current_updates, 'status': STATUS_CLOSED_EXT, 'last_update_time': time.time()}

        # --- Apply State Updates and Removals ---
        if positions_to_remove_ids or positions_to_update_state:
            new_positions_list = []
            removed_count = 0
            updated_count = 0
            for pos in self.open_positions:
                pos_id = pos.get('id')
                if pos_id in positions_to_remove_ids:
                    removed_count += 1; continue  # Skip removed positions
                if pos_id in positions_to_update_state:
                    pos.update(positions_to_update_state[pos_id])  # Apply updates
                    updated_count += 1
                new_positions_list.append(pos)

            if removed_count > 0 or updated_count > 0:
                self.open_positions = new_positions_list
                logger.debug(f"[{func_name}] Management complete. Updated: {updated_count}, Removed/Closed: {removed_count}. Total: {len(self.open_positions)}")
                self._save_state()

    def _infer_exit_reason(self, order_info: dict, position: dict, exit_price: float, current_price: float) -> str:
        """Tries to infer the reason for an external position closure."""
        pos_id = position['id']
        side = position['side']
        stored_sl = position.get('stop_loss_price')
        stored_tp = position.get('take_profit_price')
        entry_price = position['entry_price']
        reason = f"Externally Closed ({pos_id})"  # Default

        info = order_info.get('info', {})
        order_status = order_info.get('status', '').lower()
        # Bybit V5 specific fields
        bybit_status = info.get('orderStatus', '').lower()
        bybit_stop_order_type = info.get('stopOrderType', '').lower()
        bybit_close_type = info.get('closeOnTrigger', False)  # Check if it was a closing order

        # Use Bybit specific info if available
        if 'stop-loss order triggered' in bybit_status or bybit_stop_order_type == 'stoploss': reason = f"SL Triggered (Exchange: {pos_id})"
        elif 'take-profit order triggered' in bybit_status or bybit_stop_order_type == 'takeprofit': reason = f"TP Triggered (Exchange: {pos_id})"
        elif 'cancel' in bybit_status or order_status == 'canceled': reason = f"Cancelled Externally ({pos_id})"
        elif bybit_close_type: reason = f"Closed via Trigger Order ({pos_id})"  # ADL, Liquidation might also appear here?
        # Fallback to price comparison
        else:
             sl_triggered, tp_triggered = False, False
             price_tick = float(self.market_info['precision']['price'] or 0.0001)
             tolerance = max(entry_price * 0.001, price_tick * 5)  # 0.1% or 5 ticks

             if stored_sl is not None:
                  sl_triggered = (side == 'buy' and exit_price <= stored_sl + tolerance) or \
                                 (side == 'sell' and exit_price >= stored_sl - tolerance)
             if stored_tp is not None:
                  tp_triggered = (side == 'buy' and exit_price >= stored_tp - tolerance) or \
                                 (side == 'sell' and exit_price <= stored_tp + tolerance)

             if sl_triggered and tp_triggered: reason = f"SL/TP Hit (Price Ambiguous: {pos_id})"
             elif sl_triggered: reason = f"SL Hit (Inferred Price: {pos_id})"
             elif tp_triggered: reason = f"TP Hit (Inferred Price: {pos_id})"
             else: reason = f"Closed Externally (Reason Unclear: {pos_id}, Status: {order_status}/{bybit_status})"

        return reason

    def _update_trailing_stop_price(self, position: dict, current_price: float) -> float | None:
        """Calculates if TSL needs activation or update. Returns new TSL price if needed, else None."""
        func_name = "_update_trailing_stop_price"
        side = position['side']
        entry_price = position['entry_price']
        current_tsl = position.get('trailing_stop_price')
        base_sl = position.get('stop_loss_price')  # Original/Base SL
        tsl_percentage = self.trailing_stop_loss_percentage
        is_tsl_active = current_tsl is not None

        if not tsl_percentage or not (0 < tsl_percentage < 1): return None

        new_tsl_price: float | None = None
        tsl_factor = (1 - tsl_percentage) if side == 'buy' else (1 + tsl_percentage)
        potential_tsl_raw = current_price * tsl_factor

        try:
            potential_tsl = float(self.exchange.price_to_precision(self.symbol, potential_tsl_raw))
            if potential_tsl <= 0: potential_tsl = None
        except Exception as e: logger.error(f"[{func_name}] Error formatting potential TSL {potential_tsl_raw}: {e}"); potential_tsl = None
        if potential_tsl is None: return None

        # TSL Activation Check
        if not is_tsl_active:
            # Activate if price moves favorably beyond entry + buffer, AND potential TSL is better than base SL
            price_tick = float(self.market_info['precision']['price'] or 0.0001)
            activation_buffer = max(entry_price * 0.001, price_tick * 5)  # 0.1% or 5 ticks
            activation_price = entry_price + activation_buffer if side == 'buy' else entry_price - activation_buffer

            price_moved_enough = (side == 'buy' and current_price > activation_price) or \
                                 (side == 'sell' and current_price < activation_price)

            if price_moved_enough:
                # Ensure potential TSL is actually better (more protective) than the base SL (if set)
                is_better_than_base = True
                if base_sl is not None:
                    is_better_than_base = (side == 'buy' and potential_tsl > base_sl) or \
                                          (side == 'sell' and potential_tsl < base_sl)

                if is_better_than_base:
                    new_tsl_price = potential_tsl
                    logger.info(f"{Fore.MAGENTA}[{func_name}] TSL ACTIVATING for {side} {position['id']} at {self.format_price(new_tsl_price)} (Price: {self.format_price(current_price)}){Style.RESET_ALL}")
                else:
                    logger.debug(f"[{func_name}] TSL activation condition met for {position['id']}, but potential TSL {self.format_price(potential_tsl)} not better than base SL {self.format_price(base_sl)}. No activation.")
            # else: logger.debug(f"[{func_name}] Price ({self.format_price(current_price)}) not beyond activation threshold ({self.format_price(activation_price)}) for {position['id']}.")

        # TSL Update Check (if already active)
        elif is_tsl_active:
            # Update if potential TSL is better (more protective) than current TSL
            update_needed = (side == 'buy' and potential_tsl > current_tsl) or \
                            (side == 'sell' and potential_tsl < current_tsl)
            if update_needed:
                 new_tsl_price = potential_tsl
                 logger.info(f"{Fore.MAGENTA}[{func_name}] TSL UPDATING for {side} {position['id']} from {self.format_price(current_tsl)} to {self.format_price(new_tsl_price)} (Price: {self.format_price(current_price)}){Style.RESET_ALL}")

        return new_tsl_price

    def _attempt_edit_order_for_tsl(self, position: dict, new_tsl_price: float) -> bool:
        """EXPERIMENTAL: Attempts to modify order SL using `edit_order`. Returns bool indicating attempt success."""
        func_name = "_attempt_edit_order_for_tsl"
        pos_id = position['id']
        symbol = position['symbol']
        # IMPORTANT: Bybit V5 requires RESENDING TP when modifying SL via editOrder, otherwise TP gets removed.
        stored_tp_price = position.get('take_profit_price')
        logger.warning(f"{Fore.YELLOW}[EXPERIMENTAL][{func_name}] Attempting TSL update via edit_order for {pos_id}. THIS IS EXPERIMENTAL AND MAY NOT WORK RELIABLY.{Style.RESET_ALL}")

        try:
            edit_params = {
                'stopLoss': self.format_price(new_tsl_price)  # Send as string
            }
            if self.sl_trigger_by: edit_params['slTriggerBy'] = self.sl_trigger_by
            # Resend TP if it exists
            if stored_tp_price:
                edit_params['takeProfit'] = self.format_price(stored_tp_price)
                if self.tp_trigger_by: edit_params['tpTriggerBy'] = self.tp_trigger_by
                logger.debug(f"[{func_name}] Resending TP={edit_params['takeProfit']} with SL update.")
            else:
                logger.debug(f"[{func_name}] No existing TP found for {pos_id}, not resending TP.")

            # Bybit V5 category
            if self.exchange_id.lower() == 'bybit':
                market_type = self.exchange.market(symbol).get('type', 'swap')
                is_linear = self.exchange.market(symbol).get('linear', True)
                if market_type in ['swap', 'future']: edit_params['category'] = 'linear' if is_linear else 'inverse'
                elif market_type == 'spot': edit_params['category'] = 'spot'

            # edit_order generally needs original order details. Fetch FRESH info.
            fresh_order_info = self.fetch_order_status(pos_id, symbol=symbol)
            if not fresh_order_info or fresh_order_info.get('status') != 'open':
                 logger.warning(f"[{func_name}] Cannot edit TSL: Order {pos_id} not found or not 'open' (Status: {fresh_order_info.get('status')}).")
                 return False

            # Required args for edit_order (check ccxt docs for your version)
            order_type = fresh_order_info.get('type')
            order_side = fresh_order_info.get('side')
            order_amount = fresh_order_info.get('amount')  # Usually original amount
            order_price = fresh_order_info.get('price')  # Required for limit

            if not all([order_type, order_side, order_amount]): raise ValueError("Missing essential fields in fresh order info for edit.")
            if order_type == 'limit' and order_price is None: raise ValueError("Missing price in fresh limit order info for edit.")

            logger.debug(f"[{func_name}] Calling edit_order for {pos_id}. New SL: {new_tsl_price}. Params: {edit_params}")

            # Use retry decorator for the edit attempt itself
            @retry_api_call(max_retries=1, initial_delay=1)
            def _edit_call(*args, **kwargs):
                return self.exchange.edit_order(*args, **kwargs)

            edited_order = _edit_call(pos_id, symbol, order_type, order_side, order_amount, order_price, edit_params)

            if edited_order:
                confirmed_sl = edited_order.get('info', {}).get('stopLoss') or edited_order.get('stopLossPrice')
                confirmed_tp = edited_order.get('info', {}).get('takeProfit') or edited_order.get('takeProfitPrice')
                logger.info(f"{Fore.MAGENTA}---> edit_order attempt for {pos_id} complete. Exchange Response SL: {confirmed_sl}, TP: {confirmed_tp}{Style.RESET_ALL}")
                # Simple check: if edit call didn't fail hard and returned *something*, assume it *might* have worked.
                # More robust check would compare confirmed_sl with new_tsl_price.
                if confirmed_sl and abs(float(confirmed_sl) - new_tsl_price) < 1e-9:
                     logger.info(f"{Fore.GREEN}---> TSL edit CONFIRMED via response SL for {pos_id}.{Style.RESET_ALL}")
                     return True  # High confidence
                else:
                     logger.warning(f"{Fore.YELLOW}---> TSL edit response SL ({confirmed_sl}) mismatch target ({new_tsl_price}). Edit *may* have failed or confirmation lag.{Style.RESET_ALL}")
                     return False  # Low confidence, don't update internal state based on this attempt? Or maybe return True cautiously? Let's be cautious.

            else:  # edit_order returned None after retries
                logger.error(f"{Fore.RED}[{func_name}] TSL update via edit_order failed (API returned None) for {pos_id}.{Style.RESET_ALL}")
                return False

        except ccxt.NotSupported as e:
            logger.error(f"{Fore.RED}[{func_name}] TSL update FAILED: edit_order for SL/TP mod NOT SUPPORTED. Error: {e}. Disable TSL or use alternative.{Style.RESET_ALL}")
            self.enable_trailing_stop_loss = False  # Disable TSL for future runs
            return False
        except (ccxt.OrderNotFound, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
             logger.error(f"{Fore.RED}[{func_name}] TSL edit_order failed ({type(e).__name__}): {e}.{Style.RESET_ALL}")
             return False
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] Unexpected error attempting TSL edit for {pos_id}: {e}{Style.RESET_ALL}", exc_info=True)
            return False

    def _log_position_pnl(self, position: dict, exit_price: float | None, reason: str) -> None:
        """Calculates and logs PnL for a closed position."""
        func_name = "_log_position_pnl"
        pos_id = position.get('id')
        side = position.get('side')
        entry_price = position.get('entry_price')
        position_size = position.get('size')  # Should be the actual filled size
        symbol = position.get('symbol', self.symbol)

        if not all([pos_id, side, entry_price, position_size, exit_price, symbol]):
            logger.warning(f"{Fore.YELLOW}---> Pos {pos_id} Closed ({reason}). PnL calc failed: Missing data (E={entry_price}, X={exit_price}, S={position_size}).{Style.RESET_ALL}")
            return
        if entry_price <= 0 or exit_price <= 0 or position_size <= 0:
            logger.warning(f"{Fore.YELLOW}---> Pos {pos_id} Closed ({reason}). PnL calc failed: Invalid data (E={entry_price}, X={exit_price}, S={position_size}).{Style.RESET_ALL}")
            return

        try:
            pnl_quote = (exit_price - entry_price) * position_size if side == 'buy' else (entry_price - exit_price) * position_size
            pnl_pct = ((exit_price / entry_price) - 1) * 100 if side == 'buy' else ((entry_price / exit_price) - 1) * 100

            pnl_color = Fore.GREEN if pnl_quote >= 0 else Fore.RED
            quote_ccy = self.market_info['quote']
            base_ccy = self.market_info['base']

            log_msg = (f"{pnl_color}---> Position {pos_id} ({symbol}) Closed. Reason: {reason}. "
                       f"Entry: {self.format_price(entry_price)}, Exit: {self.format_price(exit_price)}, Size: {self.format_amount(position_size)} {base_ccy}. "
                       f"Est. PnL: {self.format_price(pnl_quote)} {quote_ccy} ({pnl_pct:.3f}%){Style.RESET_ALL}")
            logger.info(log_msg)

            self.daily_pnl += pnl_quote
            logger.info(f"Daily PnL Updated: {self.format_price(self.daily_pnl)} {quote_ccy}")

        except ZeroDivisionError: logger.error(f"[{func_name}] PnL calc failed for {pos_id}: Division by zero.")
        except Exception as e: logger.error(f"[{func_name}] Unexpected error calculating PnL for {pos_id}: {e}", exc_info=True)

    @retry_api_call(max_retries=1)
    def _place_market_close_order(self, position: dict[str, Any], current_price: float) -> dict[str, Any] | None:
        """Places market order to close position. Uses 'reduceOnly'. Returns order dict or None."""
        func_name = "_place_market_close_order"
        pos_id = position['id']
        side = position['side']
        size = position['size']
        symbol = position.get('symbol', self.symbol)
        base_ccy = self.market_info['base']
        self.market_info['quote']

        if size is None or size <= 1e-12:
            logger.error(f"[{func_name}] Cannot close {pos_id}: Invalid size ({size}).")
            return None

        close_side = 'sell' if side == 'buy' else 'buy'
        log_color = Fore.YELLOW

        logger.warning(f"{log_color}[{func_name}] Initiating MARKET CLOSE for {pos_id} ({symbol}). "
                       f"Entry: {side.upper()}, Close: {close_side.upper()}, Size: {self.format_amount(size)} {base_ccy})...{Style.RESET_ALL}")

        if self.simulation_mode:
            sim_close_id = f"sim_close_{int(time.time() * 1000)}_{close_side[:1]}"
            sim_avg_close = current_price
            sim_ts = int(time.time() * 1000)
            simulated_close_order = {
                "id": sim_close_id, "timestamp": sim_ts, "datetime": pd.to_datetime(sim_ts, unit='ms', utc=True).isoformat(),
                "symbol": symbol, "type": "market", "side": close_side, "status": 'closed',
                "amount": size, "filled": size, "remaining": 0.0, "average": sim_avg_close,
                "cost": size * sim_avg_close, "reduceOnly": True,
                "info": {"simulated": True, "orderId": sim_close_id, "reduceOnly": True, "closed_position_id": pos_id}
            }
            logger.info(f"{log_color}[SIMULATION] Market Close Order: ID {sim_close_id}, AvgPrice {self.format_price(sim_avg_close)}{Style.RESET_ALL}")
            return simulated_close_order

        else:  # Live Mode
            try:
                params = {}
                # Use reduceOnly - Bybit V5 uses it in params
                params['reduceOnly'] = True
                logger.debug(f"[{func_name}] Using reduceOnly=True param.")

                # Add Bybit V5 category param
                if self.exchange_id.lower() == 'bybit':
                    market_type = self.exchange.market(symbol).get('type', 'swap')
                    is_linear = self.exchange.market(symbol).get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'

                logger.debug(f"[{func_name}] Placing live market close for {pos_id}. Side: {close_side}, Size: {size}, Params: {params}")
                order = self.exchange.create_market_order(symbol, close_side, size, params=params)

                if order:
                    oid = order.get('id', 'N/A')
                    oavg = order.get('average')
                    ostatus = order.get('status', STATUS_UNKNOWN)
                    ofilled = order.get('filled', 0.0)
                    logger.info(
                        f"{log_color}---> LIVE Market Close Order Placed: ID {oid}, "
                        f"Status: {ostatus}, Filled: {self.format_amount(ofilled)}, AvgFill: {self.format_price(oavg)}{Style.RESET_ALL}"
                    )
                    return order
                else:
                    logger.error(f"{Fore.RED}[{func_name}] LIVE Market Close failed: API returned None for {pos_id}.{Style.RESET_ALL}")
                    return None

            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                 logger.error(f"{Fore.RED}[{func_name}] LIVE Close Failed for {pos_id} ({type(e).__name__}): {e}. Position already closed or size/params wrong?{Style.RESET_ALL}")
                 return None
            except ccxt.ExchangeError as e:
                 err_str = str(e).lower()
                 if "order cost not meet" in err_str or "position size is zero" in err_str or "reduce-only" in err_str or "position idx not match position mode" in err_str:
                      logger.warning(f"{Fore.YELLOW}[{func_name}] Market close for {pos_id} failed, likely already closed or reduce-only issue. Error: {e}{Style.RESET_ALL}")
                      return None  # Treat as closed potentially
                 else:
                      logger.error(f"{Fore.RED}[{func_name}] LIVE Close Failed for {pos_id} (ExchangeError): {e}{Style.RESET_ALL}")
                      return None
            except Exception as e:
                 logger.error(f"{Fore.RED}[{func_name}] LIVE Close Failed for {pos_id} (Unexpected Error): {e}{Style.RESET_ALL}", exc_info=True)
                 return None

    # --- Main Bot Execution Loop ---

    def _fetch_market_data(self) -> dict[str, Any] | None:
        """Fetches price, order book, history."""
        func_name = "_fetch_market_data"
        logger.debug(f"[{func_name}] Fetching market data bundle...")
        start_time = time.time()
        try:
            current_price = self.fetch_market_price()
            order_book_data = self.fetch_order_book()
            historical_data = self.fetch_historical_data()  # Auto-calculates lookback

            if current_price is None:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Failed to fetch current price. Skipping iteration.{Style.RESET_ALL}")
                return None
            if historical_data is None or historical_data.empty:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Failed to fetch sufficient historical data. Skipping iteration.{Style.RESET_ALL}")
                return None

            imbalance = order_book_data.get('imbalance') if order_book_data else None
            fetch_duration = time.time() - start_time
            logger.debug(f"[{func_name}] Market data fetched in {fetch_duration:.2f}s.")
            return {
                "price": current_price,
                "order_book_imbalance": imbalance,
                "historical_data": historical_data
            }
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] Unexpected error: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> dict[str, Any]:
        """Calculates all technical indicators."""
        func_name = "_calculate_indicators"
        logger.debug(f"[{func_name}] Calculating indicators...")
        start_time = time.time()
        indicators = {}
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
             logger.error(f"[{func_name}] Invalid historical data."); return {}
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in historical_data.columns for col in required_cols):
             logger.error(f"[{func_name}] History missing columns ({required_cols})."); return {}
        if len(historical_data) < 2: logger.warning(f"[{func_name}] Very short history ({len(historical_data)} rows).")

        close, high, low = historical_data['close'], historical_data['high'], historical_data['low']

        indicators['volatility'] = self.calculate_volatility(close, self.volatility_window)
        indicators['ema'] = self.calculate_ema(close, self.ema_period)
        indicators['rsi'] = self.calculate_rsi(close, self.rsi_period)
        macd_line, macd_signal, macd_hist = self.calculate_macd(close, self.macd_short_period, self.macd_long_period, self.macd_signal_period)
        indicators['macd_line'], indicators['macd_signal'], indicators['macd_hist'] = macd_line, macd_signal, macd_hist
        stoch_k, stoch_d = self.calculate_stoch_rsi(close, self.rsi_period, self.stoch_rsi_period, self.stoch_rsi_k_period, self.stoch_rsi_d_period)
        indicators['stoch_k'], indicators['stoch_d'] = stoch_k, stoch_d
        indicators['atr'] = self.calculate_atr(high, low, close, self.atr_period)

        calc_duration = time.time() - start_time
        log_msg = (
            f"Indicators (calc {calc_duration:.2f}s): "
            f"EMA={self.format_price(indicators.get('ema'))} "
            f"RSI={self.format_indicator(indicators.get('rsi'), 1)} "
            f"ATR={self.format_price(indicators.get('atr'))} "
            f"MACD={self.format_price(indicators.get('macd_line'), self.price_decimals + 1)}/{self.format_price(indicators.get('macd_signal'), self.price_decimals + 1)} "
            f"Stoch={self.format_indicator(indicators.get('stoch_k'), 1)}/{self.format_indicator(indicators.get('stoch_d'), 1)} "
            f"Vol={self.format_indicator(indicators.get('volatility'), 5)}"
        )
        logger.info(log_msg)

        nones = [k for k, v in indicators.items() if v is None]
        if nones: logger.warning(f"{Fore.YELLOW}[{func_name}] Indicators returning None: {nones}. Check data/periods.{Style.RESET_ALL}")
        return indicators

    def _process_signals_and_entry(self, market_data: dict, indicators: dict) -> None:
        """Analyzes signals, checks conditions, and potentially places entry order."""
        func_name = "_process_signals_and_entry"
        current_price = market_data['price']
        imbalance = market_data['order_book_imbalance']
        atr_value = indicators.get('atr')

        # --- 1. Check Pre-conditions ---
        active_or_pending_count = sum(1 for p in self.open_positions if p.get('status') in [STATUS_ACTIVE, STATUS_PENDING_ENTRY])
        if active_or_pending_count >= self.max_open_positions:
             logger.info(f"{Fore.CYAN}[{func_name}] Max positions ({self.max_open_positions}) reached. Skipping entry eval.{Style.RESET_ALL}")
             return

        # --- 2. Compute Signal Score ---
        signal_score, reasons = self.compute_trade_signal_score(current_price, indicators, imbalance)
        score_color = Fore.GREEN if signal_score > 0 else (Fore.RED if signal_score < 0 else Fore.WHITE)
        logger.info(f"Trade Signal Score: {score_color}{signal_score}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG) or abs(signal_score) >= ENTRY_SIGNAL_THRESHOLD_ABS:
            for reason in reasons: logger.debug(f"  -> {reason}")

        # --- 3. Determine Entry Action ---
        entry_side: str | None = None
        if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS: entry_side = 'buy'
        elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS: entry_side = 'sell'

        if not entry_side:
            if signal_score is not None: logger.info(f"[{func_name}] Neutral signal ({signal_score}). Threshold ({ENTRY_SIGNAL_THRESHOLD_ABS}) not met.")
            return

        log_color = Fore.GREEN if entry_side == 'buy' else Fore.RED
        logger.info(f"{log_color}[{func_name}] {entry_side.upper()} signal score ({signal_score}) meets threshold. Preparing entry...{Style.RESET_ALL}")

        # --- 4. Calculate Order Size ---
        # Pass indicators and score for potential adjustments
        order_size_base = self.calculate_order_size(current_price, indicators, signal_score)
        if order_size_base <= 0:
             logger.warning(f"{Fore.YELLOW}[{func_name}] Entry aborted: Calculated order size is zero or invalid (check logs from calculate_order_size).{Style.RESET_ALL}")
             return

        # --- 5. Calculate SL/TP Prices ---
        sl_price, tp_price = self._calculate_sl_tp_prices(
            entry_price=current_price,  # Use current as estimate for pre-order calc
            side=entry_side, current_price=current_price, atr=atr_value
        )
        # Abort if SL/TP calculation failed AND SL/TP is required by config
        sl_tp_required = self.use_atr_sl_tp or (self.base_stop_loss_pct is not None and self.base_take_profit_pct is not None)
        sl_tp_failed = (sl_price is None and tp_price is None)  # Check if BOTH failed
        if sl_tp_required and sl_tp_failed:
             logger.error(f"{Fore.RED}[{func_name}] Entry aborted: Required SL/TP calculation failed. Check ATR/percentage settings.{Style.RESET_ALL}")
             return
        elif sl_tp_failed:  # Calculation failed but maybe not strictly required (e.g., only SL set)
             logger.warning(f"{Fore.YELLOW}[{func_name}] SL/TP calculation failed or returned None. Proceeding without SL/TP params or only partial.{Style.RESET_ALL}")

        # --- 6. Place Entry Order ---
        entry_order = self.place_entry_order(
            side=entry_side, order_size_base=order_size_base,
            confidence_level=signal_score, order_type=self.entry_order_type,
            current_price=current_price, stop_loss_price=sl_price, take_profit_price=tp_price
        )

        # --- 7. Update Bot State ---
        if entry_order:
            order_id = entry_order.get('id')
            if not order_id:
                 logger.critical(f"{Fore.RED}[{func_name}] Entry order placed but received NO ID! Cannot track. Manual check required! Response: {str(entry_order)[:200]}{Style.RESET_ALL}")
                 return  # Cannot add position

            order_status = entry_order.get('status')
            initial_pos_status = STATUS_UNKNOWN
            is_filled_immediately = False

            if order_status == 'open': initial_pos_status = STATUS_PENDING_ENTRY
            elif order_status == 'closed': initial_pos_status = STATUS_ACTIVE; is_filled_immediately = True
            elif order_status in ['canceled', 'rejected', 'expired']:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Entry order {order_id} failed immediately (Status: {order_status}). Not adding position.{Style.RESET_ALL}")
                return
            else: initial_pos_status = STATUS_PENDING_ENTRY; logger.warning(f"[{func_name}] Entry order {order_id} unusual initial status '{order_status}'. Treating as PENDING.")

            filled_amount = float(entry_order.get('filled', 0.0))
            avg_fill_price = float(entry_order.get('average')) if entry_order.get('average') else None
            timestamp_ms = entry_order.get('timestamp')
            requested_amount = float(entry_order.get('amount', order_size_base))

            entry_price_state = avg_fill_price if is_filled_immediately and avg_fill_price else None
            entry_time_state = timestamp_ms / 1000 if is_filled_immediately and timestamp_ms else None
            size_state = filled_amount if is_filled_immediately else 0.0  # Store 0 size if pending

            new_position = {
                "id": order_id, "symbol": self.symbol, "side": entry_side,
                "size": size_state,  # Current filled size (0 if pending)
                "original_size": requested_amount,  # Requested size
                "entry_price": entry_price_state,  # None if pending
                "entry_time": entry_time_state,  # None if pending
                "status": initial_pos_status,
                "entry_order_type": self.entry_order_type,
                "stop_loss_price": sl_price,  # Store calculated SL sent
                "take_profit_price": tp_price,  # Store calculated TP sent
                "confidence": signal_score,
                "trailing_stop_price": None,
                "last_update_time": time.time()
            }
            self.open_positions.append(new_position)
            logger.info(f"{Fore.CYAN}---> Position {order_id} added to state. Status: {initial_pos_status}, Entry Price: {self.format_price(entry_price_state)}{Style.RESET_ALL}")
            self._save_state()
        else:
             logger.error(f"{Fore.RED}[{func_name}] Entry order placement failed (API call error). No position added.{Style.RESET_ALL}")

    def run(self) -> None:
        """Starts the main trading loop."""
        logger.info(f"{Fore.CYAN}--- Initiating Trading Loop (Symbol: {self.symbol}, TF: {self.timeframe}) ---{Style.RESET_ALL}")
        while True:
            self.iteration += 1
            start_time_iter = time.time()
            ts_now = pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')
            loop_prefix = f"{Fore.BLUE}===== Iter {self.iteration} ====={Style.RESET_ALL}"
            logger.info(f"\n{loop_prefix} Timestamp: {ts_now}")

            try:
                # 1. Fetch Data
                market_data = self._fetch_market_data()
                if market_data is None: time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue
                current_price = market_data['price']
                logger.info(f"{loop_prefix} Price: {self.format_price(current_price)} "
                            f"OB Imb: {self.format_indicator(market_data.get('order_book_imbalance'), 3)}")

                # 2. Calculate Indicators
                indicators = self._calculate_indicators(market_data['historical_data'])
                if not indicators: time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue

                # 3. Check Pending Entries
                self._check_pending_entries(indicators)

                # 4. Manage Active Positions (SL/TP Check, Time Exit, TSL)
                self._manage_active_positions(current_price, indicators)

                # 5. Process Signals & Potential New Entry
                self._process_signals_and_entry(market_data, indicators)

                # 6. Loop Pacing
                exec_time = time.time() - start_time_iter
                wait_time = max(0.1, DEFAULT_SLEEP_INTERVAL_SECONDS - exec_time)
                logger.debug(f"{loop_prefix} Loop took {exec_time:.2f}s. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            except KeyboardInterrupt:
                logger.warning(f"\n{Fore.YELLOW}Keyboard interrupt. Initiating shutdown...{Style.RESET_ALL}")
                break
            except SystemExit as e:
                 logger.warning(f"SystemExit called with code {e.code}. Exiting loop...")
                 raise e
            except Exception as e:
                logger.critical(f"{Fore.RED}{Style.BRIGHT}{loop_prefix} CRITICAL UNHANDLED ERROR: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
                error_pause = 60
                logger.warning(f"{Fore.YELLOW}Pausing {error_pause}s due to critical error...{Style.RESET_ALL}")
                try: time.sleep(error_pause)
                except KeyboardInterrupt: logger.warning("\nInterrupt during error pause. Exiting..."); break

        logger.info("Main trading loop terminated.")

    def shutdown(self) -> None:
        """Graceful shutdown: cancel pending, optionally close active, save state."""
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}--- Initiating Graceful Shutdown ---{Style.RESET_ALL}")
        needs_state_save = False

        # --- 1. Cancel PENDING Orders ---
        pending_to_cancel = [p for p in list(self.open_positions) if p.get('status') == STATUS_PENDING_ENTRY]
        cancelled_pending = 0
        if pending_to_cancel:
            logger.info(f"Attempting to cancel {len(pending_to_cancel)} PENDING entry order(s)...")
            for pos in pending_to_cancel:
                pos_id = pos.get('id')
                pos_symbol = pos.get('symbol', self.symbol)
                if not pos_id: continue
                logger.info(f"Cancelling pending {pos_id} ({pos_symbol})...")
                # Use the simulation status update logic from cancel_order_by_id directly
                if self.cancel_order_by_id(pos_id, symbol=pos_symbol):
                    cancelled_pending += 1
                    # Update state for cancelled order
                    try:  # Find original position to update status
                        original_pos = next(p for p in self.open_positions if p.get('id') == pos_id)
                        original_pos['status'] = STATUS_CANCELED
                        original_pos['last_update_time'] = time.time()
                        needs_state_save = True
                    except StopIteration:
                         logger.warning(f"Could not find pos {pos_id} in state list after simulated cancel success.")
                else:
                    logger.error(f"{Fore.RED}Failed to cancel pending order {pos_id} ({pos_symbol}). Manual check needed.{Style.RESET_ALL}")
            logger.info(f"Pending order cancellation finished. Cancelled/Gone: {cancelled_pending}/{len(pending_to_cancel)}.")
        else: logger.info("No pending orders in state to cancel.")

        # --- 2. Optionally Close ACTIVE Positions ---
        active_to_manage = [p for p in self.open_positions if p.get('status') == STATUS_ACTIVE]
        if self.close_positions_on_exit and active_to_manage:
             logger.warning(f"{Fore.YELLOW}Closing {len(active_to_manage)} ACTIVE position(s) due to config 'close_positions_on_exit=True'...{Style.RESET_ALL}")
             current_price = self.fetch_market_price()
             if current_price is None:
                 logger.critical(f"{Fore.RED}{Style.BRIGHT}CRITICAL: Cannot fetch price for market close! {len(active_to_manage)} positions remain OPEN! Manual action required!{Style.RESET_ALL}")
             else:
                 closed_count = 0; failed_ids = []
                 for position in list(active_to_manage):  # Iterate copy
                    pos_id = position.get('id'); pos_symbol = position.get('symbol', self.symbol)
                    logger.info(f"Attempting market close for active {pos_id} ({pos_symbol})...")
                    close_order = self._place_market_close_order(position, current_price)
                    if close_order:
                        closed_count += 1
                        logger.info(f"{Fore.YELLOW}---> Market close placed for {pos_id}. Marking as '{STATUS_CLOSED_ON_EXIT}'.{Style.RESET_ALL}")
                        position['status'] = STATUS_CLOSED_ON_EXIT  # Update the item from the *original* list
                        position['last_update_time'] = time.time()
                        needs_state_save = True
                        close_fill = close_order.get('average') or current_price
                        self._log_position_pnl(position, close_fill, f"Closed on Exit (Order {close_order.get('id')})")
                    else:
                        failed_ids.append(pos_id)
                        logger.error(f"{Fore.RED}{Style.BRIGHT}CRITICAL: Failed market close for {pos_id} ({pos_symbol}). REMAINS OPEN! Manual action required!{Style.RESET_ALL}")
                 logger.info(f"Active position closure attempt: Closed {closed_count}/{len(active_to_manage)}. Failed: {len(failed_ids)}.")
                 if failed_ids: logger.error(f"Failed close IDs: {failed_ids}. MANUAL ACTION NEEDED.")

        elif active_to_manage:
            logger.warning(f"{Fore.YELLOW}{len(active_to_manage)} position(s) remain active ('close_positions_on_exit' is false):{Style.RESET_ALL}")
            for pos in active_to_manage:
                logger.warning(f" -> ID: {pos.get('id')} ({pos.get('symbol')}), Side: {pos.get('side')}, Size: {self.format_amount(pos.get('size'))}, Entry: {self.format_price(pos.get('entry_price'))}")
        else: logger.info("No active positions found to manage during shutdown.")

        # --- 3. Final State Save ---
        if needs_state_save:
             logger.info("Saving final bot state reflecting shutdown actions...")
             self._save_state()
        else:
             logger.info("No state changes during shutdown procedure requiring save.")

        logger.info(f"{Fore.CYAN}{Style.BRIGHT}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")
        logging.shutdown()  # Ensure log handlers are flushed/closed

    # --- Helper Formatting Functions ---
    def format_price(self, price: float | int | str | None, decimals: int | None = None) -> str:
        """Safely formats price using instance default or specified decimals."""
        decs = decimals if decimals is not None else self.price_decimals
        return format_price(price, decs)

    def format_amount(self, amount: float | int | str | None, decimals: int | None = None) -> str:
        """Safely formats amount using instance default or specified decimals."""
        decs = decimals if decimals is not None else self.amount_decimals
        return format_amount(amount, decs)

    def format_indicator(self, value: float | int | str | None, precision: int = 4) -> str:
        """Safely formats indicator values."""
        if value is None: return "N/A"
        try: return f"{float(value):.{precision}f}"
        except (ValueError, TypeError): return str(value)


# --- Static Helper Functions (outside class) ---
def format_price(price: float | int | str | None, decimals: int) -> str:
    """Static helper: Safely formats price value."""
    if price is None: return "N/A"
    try: return f"{float(price):.{decimals}f}"
    except (ValueError, TypeError): return str(price)


def format_amount(amount: float | int | str | None, decimals: int) -> str:
    """Static helper: Safely formats amount value."""
    if amount is None: return "N/A"
    try: return f"{float(amount):.{decimals}f}"
    except (ValueError, TypeError): return str(amount)


# --- Main Execution Block ---
if __name__ == "__main__":
    bot_instance: ScalpingBot | None = None
    exit_code: int = 0

    try:
        # Pre-run setup: Ensure state directory exists
        state_dir = os.path.dirname(STATE_FILE_NAME)
        if state_dir and not os.path.exists(state_dir):
            try:
                os.makedirs(state_dir)
            except OSError:
                 sys.exit(1)

        # Initialize Bot (handles config, validation, exchange, state, market info)
        bot_instance = ScalpingBot(config_file=CONFIG_FILE_NAME, state_file=STATE_FILE_NAME)
        bot_instance.run()  # Start main loop

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}Shutdown signal detected (KeyboardInterrupt in main).{Style.RESET_ALL}")
        exit_code = 130
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        if exit_code not in [0, 130]: logger.error(f"Bot exited via SystemExit with code: {exit_code}.")
        else: logger.info(f"Bot exited via SystemExit with code: {exit_code}.")
    except Exception as e:
        logger.critical(f"{Fore.RED}{Style.BRIGHT}Unhandled critical error occurred outside main loop: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
        exit_code = 1
    finally:
        logger.info("Initiating final shutdown procedures...")
        if bot_instance:
            try: bot_instance.shutdown()
            except Exception:
                 logging.shutdown()  # Attempt final log shutdown
        else:
            logging.shutdown()

        sys.exit(exit_code)
