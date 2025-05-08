"""Scalping Bot v4 - Pyrmethus Enhanced Edition (Bybit V5 Optimized).

Implements an enhanced cryptocurrency scalping bot using ccxt, specifically
tailored for Bybit V5 API's parameter-based SL/TP handling.

Key Enhancements V4:
- Solidified Bybit V5 parameter-based SL/TP in create_order.
- Enhanced Trailing Stop Loss (TSL) logic using edit_order (needs verification).
- Added enable_trailing_stop_loss config flag.
- Added sl_trigger_by, tp_trigger_by config options.
- Added explicit testnet_mode config flag.
- Robust state file handling with backup on corruption.
- Improved error handling, logging, type hinting, and code clarity.
"""

import json
import logging
import os
import shutil  # For state file backup
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import ccxt

# import ccxt.async_support as ccxt_async # Keep for future async potential
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style
from colorama import init as colorama_init
from dotenv import load_dotenv

# --- Arcane Constants ---
CONFIG_FILE_NAME = "config.yaml"
STATE_FILE_NAME = "scalping_bot_state.json"
LOG_FILE_NAME = "scalping_bot_v4.log"
DEFAULT_EXCHANGE_ID = "bybit"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_RETRY_MAX = 3
DEFAULT_RETRY_DELAY_SECONDS = 3
DEFAULT_SLEEP_INTERVAL_SECONDS = 10
STRONG_SIGNAL_THRESHOLD_ABS = 3
ENTRY_SIGNAL_THRESHOLD_ABS = 2
ATR_MULTIPLIER_SL = 2.0
ATR_MULTIPLIER_TP = 3.0

# Position Status Constants
STATUS_PENDING_ENTRY = "pending_entry"
STATUS_ACTIVE = "active"
STATUS_CLOSING = "closing"  # Could be used if manual close initiated
STATUS_UNKNOWN = "unknown"

# Initialize colorama
colorama_init(autoreset=True)

# --- Logger Setup ---
logger = logging.getLogger("ScalpingBotV4")
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s"  # Added brackets
)
# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# File Handler
try:
    # Ensure log directory exists if needed (e.g., if logging to a subdir)
    # log_dir = os.path.dirname(LOG_FILE_NAME)
    # if log_dir and not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    file_handler = logging.FileHandler(LOG_FILE_NAME, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
except OSError as e:
    logger.error(f"{Fore.RED}Fatal: Failed to conjure log file {LOG_FILE_NAME}: {e}{Style.RESET_ALL}")
    file_handler = None
except Exception as e:
    logger.error(f"{Fore.RED}Fatal: Unexpected error setting up file logging: {e}{Style.RESET_ALL}")
    file_handler = None

# Load environment variables
if load_dotenv():
    logger.info(f"{Fore.CYAN}# Summoning secrets from .env scroll...{Style.RESET_ALL}")
else:
    logger.debug("No .env scroll found or secrets within.")


# --- API Retry Decorator ---
def retry_api_call(max_retries: int = DEFAULT_RETRY_MAX, initial_delay: int = DEFAULT_RETRY_DELAY_SECONDS) -> Callable:
    """Decorator for retrying CCXT API calls with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any | None:
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    # Channeling the API call...
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(
                        f"{Fore.YELLOW}Rate limit veil encountered ({func.__name__}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}) Error: {e}{Style.RESET_ALL}"
                    )
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                    logger.error(
                        f"{Fore.RED}Network ether disturbed ({func.__name__}: {type(e).__name__}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # Non-retryable conditions
                    if any(
                        phrase in err_str
                        for phrase in [
                            "order not found",
                            "order does not exist",
                            "unknown order",
                            "order already cancelled",
                        ]
                    ):
                        logger.warning(
                            f"{Fore.YELLOW}Order vanished or unknown for {func.__name__}: {e}. Returning None (no retry).{Style.RESET_ALL}"
                        )
                        return None  # OrderNotFound is often final
                    elif any(
                        phrase in err_str
                        for phrase in ["insufficient balance", "insufficient margin", "margin is insufficient"]
                    ):
                        logger.error(
                            f"{Fore.RED}Insufficient essence (funds/margin) for {func.__name__}: {e}. Aborting (no retry).{Style.RESET_ALL}"
                        )
                        return None
                    elif any(
                        phrase in err_str
                        for phrase in [
                            "invalid order",
                            "parameter error",
                            "size too small",
                            "price too low",
                            "price too high",
                        ]
                    ):
                        logger.error(
                            f"{Fore.RED}Invalid parameters/limits for {func.__name__}: {e}. Aborting (no retry). Check logic/config.{Style.RESET_ALL}"
                        )
                        return None
                    else:  # Retry other exchange errors
                        logger.error(
                            f"{Fore.RED}Exchange spirit troubled ({func.__name__}: {e}). Pausing {delay}s... "
                            f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                        )
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Unexpected rift during {func.__name__}: {type(e).__name__} - {e}. Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}",
                        exc_info=True,
                    )

                # Wait and increase delay if retry is needed
                if retries < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    retries += 1
                else:
                    logger.error(
                        f"{Fore.RED}Max retries ({max_retries}) reached for {func.__name__}. The spell falters.{Style.RESET_ALL}"
                    )
                    return None  # Indicate failure

            return None  # Should not be reached

        return wrapper

    return decorator


# --- Scalping Bot Class ---
class ScalpingBot:
    """Pyrmethus Enhanced Scalping Bot v4. Bybit V5 Optimized.
    Features persistent state, ATR-based SL/TP, parameter-based SL/TP placement,
    and refined order management with TSL via edit_order (verification required).
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME, state_file: str = STATE_FILE_NAME) -> None:
        """Initializes the bot, loading config, state, and connecting to the exchange."""
        logger.info(f"{Fore.MAGENTA}--- Pyrmethus Scalping Bot v4 Awakening ---{Style.RESET_ALL}")
        self.config: dict[str, Any] = {}
        self.state_file: str = state_file
        self.load_config(config_file)
        self.validate_config()

        # --- Bind Attributes from Config/Env ---
        self.api_key: str | None = os.getenv("BYBIT_API_KEY")
        self.api_secret: str | None = os.getenv("BYBIT_API_SECRET")
        self.exchange_id: str = self.config["exchange"]["exchange_id"]
        self.testnet_mode: bool = self.config["exchange"]["testnet_mode"]  # New explicit flag
        self.symbol: str = self.config["trading"]["symbol"]
        self.simulation_mode: bool = self.config["trading"]["simulation_mode"]  # Keep for internal logic simulation
        self.entry_order_type: str = self.config["trading"]["entry_order_type"]
        self.limit_order_entry_offset_pct_buy: float = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_entry_offset_pct_sell: float = self.config["trading"]["limit_order_offset_sell"]
        self.timeframe: str = self.config["trading"]["timeframe"]
        self.order_book_depth: int = self.config["order_book"]["depth"]
        self.imbalance_threshold: float = self.config["order_book"]["imbalance_threshold"]
        # Indicators
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
        # Risk Management
        self.use_atr_sl_tp: bool = self.config["risk_management"]["use_atr_sl_tp"]
        self.atr_sl_multiplier: float = self.config["risk_management"]["atr_sl_multiplier"]
        self.atr_tp_multiplier: float = self.config["risk_management"]["atr_tp_multiplier"]
        self.base_stop_loss_pct: float | None = self.config["risk_management"].get("stop_loss_percentage")
        self.base_take_profit_pct: float | None = self.config["risk_management"].get("take_profit_percentage")
        self.sl_trigger_by: str | None = self.config["risk_management"].get(
            "sl_trigger_by"
        )  # e.g., MarkPrice, LastPrice
        self.tp_trigger_by: str | None = self.config["risk_management"].get(
            "tp_trigger_by"
        )  # e.g., MarkPrice, LastPrice
        self.enable_trailing_stop_loss: bool = self.config["risk_management"][
            "enable_trailing_stop_loss"
        ]  # New TSL flag
        self.trailing_stop_loss_percentage: float | None = self.config["risk_management"].get(
            "trailing_stop_loss_percentage"
        )
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.time_based_exit_minutes: int | None = self.config["risk_management"].get("time_based_exit_minutes")
        self.order_size_percentage: float = self.config["risk_management"]["order_size_percentage"]
        # Signal adjustments (kept simple for now)
        self.strong_signal_adjustment_factor: float = self.config["risk_management"]["strong_signal_adjustment_factor"]
        self.weak_signal_adjustment_factor: float = self.config["risk_management"]["weak_signal_adjustment_factor"]

        # --- Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0
        # Stores active and pending positions: List[Dict[str, Any]]
        # Structure: {'id': entry_order_id, 'side': 'buy'/'sell', 'size': float,
        #             'entry_price': float, 'entry_time': float, 'status': STATUS_*,
        #             'entry_order_type': str, 'stop_loss_price': float|None,
        #             'take_profit_price': float|None, 'confidence': int,
        #             'trailing_stop_price': float|None, 'last_update_time': float }
        # Note: sl_order_id and tp_order_id removed for Bybit V5 parameter approach
        self.open_positions: list[dict[str, Any]] = []
        self.market_info: dict[str, Any] | None = None  # Cache market info

        self._configure_logging_level()
        self.exchange: ccxt.Exchange = self._initialize_exchange()

        # --- Load Market Info ---
        self._load_market_info()  # Load and cache market details

        # --- Load Persistent State ---
        self._load_state()

        # --- Final Mode Logging ---
        sim_color = Fore.YELLOW if self.simulation_mode else Fore.CYAN
        test_color = Fore.YELLOW if self.testnet_mode else Fore.GREEN
        logger.warning(f"{sim_color}--- INTERNAL SIMULATION MODE: {self.simulation_mode} ---{Style.RESET_ALL}")
        logger.warning(f"{test_color}--- EXCHANGE TESTNET MODE: {self.testnet_mode} ---{Style.RESET_ALL}")
        if not self.simulation_mode and not self.testnet_mode:
            logger.warning(f"{Fore.RED}--- LIVE TRADING ON MAINNET ACTIVE ---{Style.RESET_ALL}")
        elif not self.simulation_mode and self.testnet_mode:
            logger.warning(f"{Fore.YELLOW}--- LIVE TRADING ON TESTNET ACTIVE ---{Style.RESET_ALL}")

        logger.info(
            f"{Fore.CYAN}Scalping Bot V4 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}"
        )

    def _configure_logging_level(self) -> None:
        """Sets logger level based on config."""
        log_level_str = self.config.get("logging_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        # Set root logger level FIRST if you want handlers to inherit it by default
        logging.getLogger().setLevel(logging.DEBUG)  # Set root to lowest level
        # Then set specific handler levels
        logger.setLevel(log_level)  # Set our bot's logger level
        console_handler.setLevel(log_level)  # Console mirrors bot logger level
        if file_handler:
            file_handler.setLevel(logging.DEBUG)  # File log remains detailed
        logger.info(f"Logging level enchanted to: {log_level_str}")
        logger.debug(f"Root logger level: {logging.getLevelName(logging.getLogger().level)}")
        logger.debug(f"Bot logger level: {logging.getLevelName(logger.level)}")
        logger.debug(f"Console handler level: {logging.getLevelName(console_handler.level)}")
        if file_handler:
            logger.debug(f"File handler level: {logging.getLevelName(file_handler.level)}")

    def _load_state(self) -> None:
        """Loads bot state from file, with backup on corruption."""
        logger.debug(f"Attempting to recall state from {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():  # Check if file is empty or just whitespace
                        logger.warning(
                            f"{Fore.YELLOW}State file {self.state_file} is empty. Starting fresh.{Style.RESET_ALL}"
                        )
                        self.open_positions = []
                        return

                    saved_state = json.loads(content)
                    if isinstance(saved_state, list):
                        # Optional: Add more rigorous validation of each position dict here
                        self.open_positions = saved_state
                        logger.info(
                            f"{Fore.GREEN}Recalled {len(self.open_positions)} position(s) from state file.{Style.RESET_ALL}"
                        )
                        # Remove backup if load was successful
                        if os.path.exists(state_backup_file):
                            try:
                                os.remove(state_backup_file)
                            except OSError:
                                logger.warning(f"Could not remove old state backup file: {state_backup_file}")
                    else:
                        raise ValueError("Invalid state format - expected a list.")
            else:
                logger.info(f"No prior state file found ({self.state_file}). Beginning anew.")
                self.open_positions = []
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"{Fore.RED}Error decoding state file {self.state_file}: {e}. Attempting to restore backup or start fresh.{Style.RESET_ALL}"
            )
            # Try restoring backup
            if os.path.exists(state_backup_file):
                try:
                    shutil.copyfile(state_backup_file, self.state_file)
                    logger.warning(
                        f"{Fore.YELLOW}Restored state from backup file {state_backup_file}. Retrying load...{Style.RESET_ALL}"
                    )
                    self._load_state()  # Retry loading after restoring backup
                    return  # Exit this attempt, the retry will handle it
                except Exception as backup_err:
                    logger.error(
                        f"{Fore.RED}Failed to restore state from backup {state_backup_file}: {backup_err}. Starting fresh.{Style.RESET_ALL}"
                    )
                    self.open_positions = []
                    self._save_state()  # Create a fresh state file
            else:
                # No backup exists, create backup of corrupted file and start fresh
                try:
                    shutil.copyfile(self.state_file, f"{self.state_file}.corrupted_{int(time.time())}")
                    logger.warning(
                        f"{Fore.YELLOW}Backed up corrupted state file to {self.state_file}.corrupted_... Starting fresh.{Style.RESET_ALL}"
                    )
                except Exception as backup_err:
                    logger.error(f"{Fore.RED}Could not back up corrupted state file: {backup_err}{Style.RESET_ALL}")
                self.open_positions = []
                self._save_state()  # Create a fresh state file
        except Exception as e:
            logger.error(
                f"{Fore.RED}Fatal: Failed to load state from {self.state_file}: {e}{Style.RESET_ALL}", exc_info=True
            )
            logger.warning("Proceeding with empty state due to unexpected load failure.")
            self.open_positions = []

    def _save_state(self) -> None:
        """Saves the current bot state to the state file."""
        # Avoid saving empty state if initialization failed badly? Maybe not necessary.
        logger.debug(f"Recording current state ({len(self.open_positions)} positions) to {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        try:
            # Create backup before overwriting
            if os.path.exists(self.state_file):
                shutil.copyfile(self.state_file, state_backup_file)

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.open_positions, f, indent=4, default=str
                )  # Use default=str for non-serializable types if any sneak in
            logger.debug("State recorded successfully.")
        except OSError as e:
            logger.error(f"{Fore.RED}Could not scribe state to {self.state_file}: {e}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(
                f"{Fore.RED}An unexpected error occurred while recording state: {e}{Style.RESET_ALL}", exc_info=True
            )

    def load_config(self, config_file: str) -> None:
        """Loads configuration from YAML or creates a default one."""
        try:
            with open(config_file, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict):
                logger.critical(f"{Fore.RED}Config file {config_file} structure invalid. Aborting.{Style.RESET_ALL}")
                sys.exit(1)
            logger.info(f"{Fore.GREEN}Configuration spellbook loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.warning(f"{Fore.YELLOW}Configuration spellbook '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
                logger.info(
                    f"{Fore.YELLOW}Crafted a default spellbook: '{config_file}'. "
                    f"Review and tailor its enchantments (API keys, symbol, risk settings), then restart.{Style.RESET_ALL}"
                )
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to craft default spellbook: {e}{Style.RESET_ALL}")
            sys.exit(1)  # Exit after creating default config
        except yaml.YAMLError as e:
            logger.critical(
                f"{Fore.RED}Error parsing spellbook {config_file}: {e}. Check syntax. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Unexpected chaos loading config: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file."""
        default_config = {
            "logging_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
            "exchange": {
                "exchange_id": os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE_ID),
                "testnet_mode": os.getenv("TESTNET_MODE", "True").lower()
                in ("true", "1", "yes"),  # Explicit testnet flag
            },
            "trading": {
                "symbol": os.getenv("TRADING_SYMBOL", "BTC/USDT:USDT"),
                "timeframe": os.getenv("TIMEFRAME", DEFAULT_TIMEFRAME),
                "simulation_mode": os.getenv("SIMULATION_MODE", "True").lower()
                in ("true", "1", "yes"),  # Internal simulation
                "entry_order_type": os.getenv("ENTRY_ORDER_TYPE", "limit").lower(),  # 'limit' or 'market'
                "limit_order_offset_buy": float(os.getenv("LIMIT_ORDER_OFFSET_BUY", 0.0005)),
                "limit_order_offset_sell": float(os.getenv("LIMIT_ORDER_OFFSET_SELL", 0.0005)),
            },
            "order_book": {"depth": 10, "imbalance_threshold": 1.5},
            "indicators": {
                "volatility_window": 20,
                "volatility_multiplier": 0.0,
                "ema_period": 10,
                "rsi_period": 14,
                "macd_short_period": 12,
                "macd_long_period": 26,
                "macd_signal_period": 9,
                "stoch_rsi_period": 14,
                "stoch_rsi_k_period": 3,
                "stoch_rsi_d_period": 3,
                "atr_period": 14,
            },
            "risk_management": {
                "order_size_percentage": 0.01,  # 1% of free quote balance
                "max_open_positions": 1,
                "use_atr_sl_tp": True,  # Use ATR or fixed %
                "atr_sl_multiplier": ATR_MULTIPLIER_SL,
                "atr_tp_multiplier": ATR_MULTIPLIER_TP,
                "stop_loss_percentage": 0.005,
                "take_profit_percentage": 0.01,  # Used if use_atr_sl_tp is false
                # Optional: Specify trigger price type for SL/TP (e.g., 'MarkPrice', 'LastPrice', 'IndexPrice') - Check Bybit/CCXT docs
                "sl_trigger_by": "MarkPrice",  # Default to MarkPrice, adjust as needed
                "tp_trigger_by": "MarkPrice",  # Default to MarkPrice, adjust as needed
                "enable_trailing_stop_loss": True,  # Enable/disable TSL feature
                "trailing_stop_loss_percentage": 0.003,  # 0.3% trail (used if TSL enabled)
                "time_based_exit_minutes": 60,  # Set 0 to disable
                "strong_signal_adjustment_factor": 1.0,  # 1.0 = no adjustment
                "weak_signal_adjustment_factor": 1.0,  # 1.0 = no adjustment
            },
        }
        # --- Auto-fill from Env Vars ---
        # (This part could be more sophisticated, iterating through nested dicts)
        if os.getenv("ORDER_SIZE_PERCENTAGE"):
            default_config["risk_management"]["order_size_percentage"] = float(os.environ["ORDER_SIZE_PERCENTAGE"])
        # ... add more overrides from env if desired ...

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(default_config, f, indent=4, sort_keys=False, default_flow_style=False)
            logger.info(f"Default spellbook '{config_file}' crafted.")
        except OSError as e:
            logger.error(f"{Fore.RED}Could not scribe default spellbook {config_file}: {e}{Style.RESET_ALL}")
            raise

    def validate_config(self) -> None:
        """Validates the loaded configuration."""
        logger.debug("Scrutinizing the configuration spellbook...")
        try:
            # --- Helpers ---
            def is_positive(v):
                return isinstance(v, (int, float)) and v > 0

            def is_non_negative(v):
                return isinstance(v, (int, float)) and v >= 0

            def is_percentage(v):
                return isinstance(v, (int, float)) and 0 <= v <= 1

            def is_string(v):
                return isinstance(v, str) and v

            def is_bool(v):
                return isinstance(v, bool)

            def is_pos_int(v):
                return isinstance(v, int) and v > 0

            def is_non_neg_int(v):
                return isinstance(v, int) and v >= 0

            # --- Structure ---
            req = {"exchange", "trading", "order_book", "indicators", "risk_management"}
            if not req.issubset(self.config.keys()):
                raise ValueError(f"Missing sections: {req - self.config.keys()}")
            for s in req:
                if not isinstance(self.config[s], dict):
                    raise ValueError(f"Section '{s}' must be a dictionary.")

            # --- Exchange ---
            cfg = self.config["exchange"]
            if not is_string(cfg.get("exchange_id")):
                raise ValueError("exchange.exchange_id required")
            if cfg["exchange_id"] not in ccxt.exchanges:
                raise ValueError(f"Exchange '{cfg['exchange_id']}' unknown")
            if not is_bool(cfg.get("testnet_mode")):
                raise ValueError("exchange.testnet_mode must be boolean")

            # --- Trading ---
            cfg = self.config["trading"]
            if not is_string(cfg.get("symbol")):
                raise ValueError("trading.symbol required")
            if not is_string(cfg.get("timeframe")):
                raise ValueError("trading.timeframe required")
            if not is_bool(cfg.get("simulation_mode")):
                raise ValueError("trading.simulation_mode must be boolean")
            if cfg.get("entry_order_type") not in ["market", "limit"]:
                raise ValueError("trading.entry_order_type must be 'market' or 'limit'")
            if not is_non_negative(cfg.get("limit_order_offset_buy")):
                raise ValueError("trading.limit_order_offset_buy must be non-negative")
            if not is_non_negative(cfg.get("limit_order_offset_sell")):
                raise ValueError("trading.limit_order_offset_sell must be non-negative")

            # --- Order Book ---
            cfg = self.config["order_book"]
            if not is_pos_int(cfg.get("depth")):
                raise ValueError("order_book.depth must be positive integer")
            if not is_positive(cfg.get("imbalance_threshold")):
                raise ValueError("order_book.imbalance_threshold must be positive")

            # --- Indicators ---
            cfg = self.config["indicators"]
            periods = [
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
            for p in periods:
                if not is_pos_int(cfg.get(p)):
                    raise ValueError(f"indicators.{p} must be positive integer")
            if not is_non_negative(cfg.get("volatility_multiplier")):
                raise ValueError("indicators.volatility_multiplier non-negative")
            if cfg["macd_short_period"] >= cfg["macd_long_period"]:
                raise ValueError("indicators.macd_short_period < macd_long_period required")

            # --- Risk Management ---
            cfg = self.config["risk_management"]
            if not is_bool(cfg.get("use_atr_sl_tp")):
                raise ValueError("risk.use_atr_sl_tp must be boolean")
            if cfg["use_atr_sl_tp"]:
                if not is_positive(cfg.get("atr_sl_multiplier")):
                    raise ValueError("risk.atr_sl_multiplier positive required if using ATR")
                if not is_positive(cfg.get("atr_tp_multiplier")):
                    raise ValueError("risk.atr_tp_multiplier positive required if using ATR")
            else:
                if not is_percentage(cfg.get("stop_loss_percentage")):
                    raise ValueError("risk.stop_loss_percentage (0-1) required if not using ATR")
                if not is_percentage(cfg.get("take_profit_percentage")):
                    raise ValueError("risk.take_profit_percentage (0-1) required if not using ATR")

            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None]  # Allow None if not specified
            if cfg.get("sl_trigger_by") not in valid_triggers:
                raise ValueError(f"risk.sl_trigger_by invalid, use: {valid_triggers}")
            if cfg.get("tp_trigger_by") not in valid_triggers:
                raise ValueError(f"risk.tp_trigger_by invalid, use: {valid_triggers}")

            if not is_bool(cfg.get("enable_trailing_stop_loss")):
                raise ValueError("risk.enable_trailing_stop_loss must be boolean")
            if cfg["enable_trailing_stop_loss"]:
                tsl_pct = cfg.get("trailing_stop_loss_percentage")
                if not (is_positive(tsl_pct) and tsl_pct < 1):
                    raise ValueError("risk.trailing_stop_loss_percentage (0 < pct < 1) required if TSL enabled")

            if not is_percentage(cfg.get("order_size_percentage")):
                raise ValueError("risk.order_size_percentage (0-1) required")
            if not is_pos_int(cfg.get("max_open_positions")):
                raise ValueError("risk.max_open_positions >= 1 required")
            if not is_non_neg_int(cfg.get("time_based_exit_minutes")):
                raise ValueError("risk.time_based_exit_minutes >= 0 required")
            if not is_positive(cfg.get("strong_signal_adjustment_factor")):
                raise ValueError("risk.strong_signal_adjustment_factor positive required")
            if not is_positive(cfg.get("weak_signal_adjustment_factor")):
                raise ValueError("risk.weak_signal_adjustment_factor positive required")

            logger.info(f"{Fore.GREEN}Configuration spellbook deemed valid and potent.{Style.RESET_ALL}")

        except ValueError as e:
            logger.critical(
                f"{Fore.RED}Configuration flaw detected: {e}. Mend the '{CONFIG_FILE_NAME}' scroll. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Unexpected chaos during config validation: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and connects to the specified exchange."""
        logger.info(f"Opening communication channel with {self.exchange_id.upper()}...")
        creds_found = self.api_key and self.api_secret
        if not self.simulation_mode and not creds_found:
            logger.critical(
                f"{Fore.RED}API Key/Secret essence missing. Cannot trade live/testnet. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        elif not creds_found:
            logger.warning(
                f"{Fore.YELLOW}API Key/Secret missing. Running in full simulation mode only.{Style.RESET_ALL}"
            )
            self.simulation_mode = True  # Force simulation if no keys, regardless of config

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange_config = {
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
            }
            # Add keys only if not in internal simulation mode
            if not self.simulation_mode:
                exchange_config["apiKey"] = self.api_key
                exchange_config["secret"] = self.api_secret

            exchange = exchange_class(exchange_config)

            # Set testnet mode based on config flag (only if not in internal simulation)
            if not self.simulation_mode and self.testnet_mode:
                logger.info("Attempting to switch exchange instance to Testnet mode...")
                exchange.set_sandbox_mode(True)  # Use CCXT's unified method

            logger.debug("Loading market matrix...")
            exchange.load_markets(True)  # Force reload
            logger.debug("Market matrix loaded.")

            # Validate timeframe and symbol after loading markets
            if self.timeframe not in exchange.timeframes:
                available = list(exchange.timeframes.keys())
                logger.critical(
                    f"{Fore.RED}Timeframe '{self.timeframe}' not supported by {self.exchange_id}. Available: {available[:15]}... Aborting.{Style.RESET_ALL}"
                )
                sys.exit(1)
            if self.symbol not in exchange.markets:
                available = list(exchange.markets.keys())
                logger.critical(
                    f"{Fore.RED}Symbol '{self.symbol}' not found on {self.exchange_id}. Available: {available[:10]}... Check config. Aborting.{Style.RESET_ALL}"
                )
                sys.exit(1)
            logger.info(f"Symbol '{self.symbol}' and timeframe '{self.timeframe}' confirmed.")

            # Final connection check
            try:
                server_time = exchange.fetch_time()
                logger.debug(f"Exchange time crystal synchronized: {pd.to_datetime(server_time, unit='ms', utc=True)}")
            except Exception as e:
                logger.warning(
                    f"{Fore.YELLOW}Optional check: Failed to fetch server time, proceeding cautiously: {e}{Style.RESET_ALL}"
                )

            logger.info(f"{Fore.GREEN}Connection established with {self.exchange_id.upper()}.{Style.RESET_ALL}")
            return exchange

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED}Authentication failed for {self.exchange_id.upper()}: {e}. Check API keys/permissions. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError, ccxt.DDoSProtection) as e:
            logger.critical(
                f"{Fore.RED}Could not connect to {self.exchange_id.upper()} ({type(e).__name__}): {e}. Check network/exchange status. Aborting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}Unexpected error initializing exchange {self.exchange_id.upper()}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> None:
        """Loads and caches market information for the symbol."""
        logger.debug(f"Loading market details for {self.symbol}...")
        try:
            self.market_info = self.exchange.market(self.symbol)
            # Pre-validate required precision/limits info
            if not self.market_info.get("precision", {}).get("amount") or not self.market_info.get("precision", {}).get(
                "price"
            ):
                raise ValueError("Amount or Price precision missing in market info.")
            # Log key limits for reference
            limits = self.market_info.get("limits", {})
            min_amount = limits.get("amount", {}).get("min")
            min_cost = limits.get("cost", {}).get("min")
            logger.info(f"Market Limits for {self.symbol}: Min Amount={min_amount}, Min Cost={min_cost}")
            logger.debug("Market details loaded and cached.")
        except Exception as e:
            logger.critical(
                f"{Fore.RED}Failed to load crucial market info for {self.symbol}: {e}. Cannot continue. Aborting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    # --- Data Fetching Methods ---
    # (fetch_market_price, fetch_order_book, fetch_historical_data, fetch_balance, fetch_order_status remain mostly the same as V3 Bybit-mod)
    # Add minor logging improvements or error checks if needed.

    @retry_api_call()
    def fetch_market_price(self) -> float | None:
        """Fetches the last traded price."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        if ticker and ticker.get("last") is not None:
            price = float(ticker["last"])
            logger.debug(
                f"Current market price ({self.symbol}): {price:.{self.market_info['precision']['price']}f}"
            )  # Use market precision
            return price
        else:
            logger.warning(
                f"{Fore.YELLOW}Could not fetch valid 'last' price for {self.symbol}. Ticker: {ticker}{Style.RESET_ALL}"
            )
            return None

    @retry_api_call()
    def fetch_order_book(self) -> dict[str, Any] | None:
        """Fetches order book data (bids, asks, imbalance)."""
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            result = {"bids": bids, "asks": asks, "imbalance": None}

            if bids and asks:
                bid_volume = sum(float(bid[1]) for bid in bids if bid and len(bid) > 1 and bid[1] is not None)
                ask_volume = sum(float(ask[1]) for ask in asks if ask and len(ask) > 1 and ask[1] is not None)

                if bid_volume > 1e-9:
                    imbalance_ratio = ask_volume / bid_volume
                    result["imbalance"] = imbalance_ratio
                    logger.debug(f"Order Book ({self.symbol}) Imbalance (Ask/Bid): {imbalance_ratio:.3f}")
                elif ask_volume > 1e-9:
                    result["imbalance"] = float("inf")
                    logger.debug(f"Order Book ({self.symbol}) Imbalance: Infinity (Zero Bid Vol)")
                else:
                    logger.debug(f"Order Book ({self.symbol}) Imbalance: N/A (Zero Bid/Ask Vol)")
            else:
                logger.warning(
                    f"{Fore.YELLOW}Order book data incomplete for {self.symbol}. Bids: {len(bids)}, Asks: {len(asks)}{Style.RESET_ALL}"
                )
            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching order book for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return None  # Return None on failure

    @retry_api_call(max_retries=2, initial_delay=1)
    def fetch_historical_data(self, limit: int | None = None) -> pd.DataFrame | None:
        """Fetches and prepares historical OHLCV data."""
        required_limit = limit
        if required_limit is None:
            required_limit = (
                max(
                    self.volatility_window + 1,
                    self.ema_period,
                    self.rsi_period + 1,
                    self.macd_long_period + self.macd_signal_period,
                    self.stoch_rsi_period + self.stoch_rsi_k_period + self.stoch_rsi_d_period + 1,
                    self.atr_period + 1,
                )
                + 20
            )  # Buffer

        logger.debug(f"Fetching {required_limit} historical candles for {self.symbol} ({self.timeframe})...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=required_limit)
            if not ohlcv:
                logger.warning(f"{Fore.YELLOW}No historical data returned for {self.symbol}.{Style.RESET_ALL}")
                return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)  # Ensure UTC
            df.set_index("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            initial_len = len(df)
            df.dropna(inplace=True)
            if len(df) < initial_len:
                logger.debug(f"Dropped {initial_len - len(df)} rows with NaNs from history.")

            min_required_len = max(self.rsi_period + 1, self.macd_long_period, self.atr_period + 1)
            if len(df) < min_required_len:
                logger.warning(
                    f"{Fore.YELLOW}Insufficient history after cleaning for {self.symbol}. Got {len(df)}, needed ~{min_required_len}. Indicators may fail.{Style.RESET_ALL}"
                )
                return None

            logger.debug(f"Fetched and processed {len(df)} historical candles for {self.symbol}.")
            return df

        except Exception as e:
            logger.error(
                f"{Fore.RED}Error fetching/processing history for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True
            )
            return None

    @retry_api_call()
    def fetch_balance(self, currency_code: str | None = None) -> float:
        """Fetches the free balance for a currency."""
        quote_currency = currency_code or self.market_info["quote"]
        logger.debug(f"Fetching balance for {quote_currency}...")
        try:
            balance_data = self.exchange.fetch_balance()
            # Use 'free' primarily, but check 'usable' for Bybit V5 unified margin if 'free' is zero/None
            free_balance = balance_data.get(quote_currency, {}).get("free")
            if free_balance is None or free_balance == 0.0:
                usable_balance = (
                    balance_data.get("info", {}).get("result", {}).get("list", [{}])[0].get("totalAvailableBalance")
                )  # Example for Bybit V5 Unified
                if usable_balance is not None:
                    logger.debug(
                        f"Using 'totalAvailableBalance' ({usable_balance}) as free balance for {quote_currency}"
                    )
                    free_balance = usable_balance

            free_balance = float(free_balance) if free_balance is not None else 0.0
            logger.info(f"Fetched free balance: {free_balance:.4f} {quote_currency}")
            return free_balance

        except ccxt.AuthenticationError:
            if self.simulation_mode:
                logger.warning(
                    f"{Fore.YELLOW}Simulation mode: Returning dummy balance 10000 {quote_currency}.{Style.RESET_ALL}"
                )
                return 10000.0
            else:
                logger.error(f"{Fore.RED}Authentication failed fetching balance for {quote_currency}.{Style.RESET_ALL}")
                return 0.0
        except Exception as e:
            logger.error(f"{Fore.RED}Could not fetch balance for {quote_currency}: {e}{Style.RESET_ALL}", exc_info=True)
            return 0.0

    @retry_api_call(max_retries=1)
    def fetch_order_status(self, order_id: str, symbol: str | None = None) -> dict[str, Any] | None:
        """Fetches the status of a specific order by its ID."""
        if not order_id:
            return None  # No ID to fetch
        target_symbol = symbol or self.symbol
        logger.debug(f"Fetching status for order {order_id} ({target_symbol})...")
        try:
            order_info = self.exchange.fetch_order(order_id, target_symbol)
            # Log essential status info
            status = order_info.get("status", STATUS_UNKNOWN)
            filled = order_info.get("filled", 0.0)
            avg_price = order_info.get("average")
            logger.debug(f"Order {order_id} status: {status}, Filled: {filled}, AvgPrice: {avg_price}")
            return order_info
        except ccxt.OrderNotFound:
            logger.warning(
                f"{Fore.YELLOW}Order {order_id} not found on exchange ({target_symbol}). Assumed closed/cancelled.{Style.RESET_ALL}"
            )
            return None  # Return None, let calling function decide how to handle
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching status for order {order_id}: {e}{Style.RESET_ALL}")
            # Don't log full trace usually, retry handles network/rate limits
            return None  # Status is unknown

    # --- Indicator Calculation Methods ---
    # (calculate_volatility, calculate_ema, calculate_rsi, calculate_macd, calculate_stoch_rsi, calculate_atr remain same as V3 Bybit-mod)
    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> float | None:
        if close_prices is None or len(close_prices) < window + 1:
            return None
        try:
            log_returns = np.log(close_prices / close_prices.shift(1))
            volatility = log_returns.rolling(window=window).std().iloc[-1]
            return float(volatility) if pd.notna(volatility) else None
        except Exception:
            return None

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> float | None:
        if close_prices is None or len(close_prices) < period:
            return None
        try:
            ema = close_prices.ewm(span=period, adjust=False).mean().iloc[-1]
            return float(ema) if pd.notna(ema) else None
        except Exception:
            return None

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> float | None:
        if close_prices is None or len(close_prices) < period + 1:
            return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
            loss = -delta.where(delta < 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        except Exception:
            return None

    @staticmethod
    def calculate_macd(
        close_prices: pd.Series, short_period: int, long_period: int, signal_period: int
    ) -> tuple[float | None, float | None, float | None]:
        min_len = long_period + signal_period
        if close_prices is None or len(close_prices) < min_len:
            return None, None, None
        try:
            ema_short = close_prices.ewm(span=short_period, adjust=False).mean()
            ema_long = close_prices.ewm(span=long_period, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            macd_val, signal_val, hist_val = macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
            if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
                return None, None, None
            return float(macd_val), float(signal_val), float(hist_val)
        except Exception:
            return None, None, None

    @staticmethod
    def calculate_stoch_rsi(
        close_prices: pd.Series, rsi_period: int, stoch_period: int, k_period: int, d_period: int
    ) -> tuple[float | None, float | None]:
        min_len = rsi_period + stoch_period + max(k_period, d_period)
        if close_prices is None or len(close_prices) < min_len:
            return None, None
        try:
            rsi = ScalpingBot.calculate_rsi(close_prices, rsi_period)  # Use static call
            if rsi is None:
                return None, None  # Need RSI first
            # Need series for rolling calculation, re-calculate RSI series here
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / rsi_period, adjust=False).mean()
            loss = -delta.where(delta < 0, 0.0).ewm(alpha=1 / rsi_period, adjust=False).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna()
            if len(rsi_series) < stoch_period:
                return None, None

            min_rsi = rsi_series.rolling(window=stoch_period).min()
            max_rsi = rsi_series.rolling(window=stoch_period).max()
            stoch_rsi = (100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)).dropna()
            if len(stoch_rsi) < max(k_period, d_period):
                return None, None

            stoch_k = stoch_rsi.rolling(window=k_period).mean()
            stoch_d = stoch_k.rolling(window=d_period).mean()
            k_val, d_val = stoch_k.iloc[-1], stoch_d.iloc[-1]
            if pd.isna(k_val) or pd.isna(d_val):
                return None, None
            return float(k_val), float(d_val)
        except Exception:
            return None, None

    @staticmethod
    def calculate_atr(
        high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int
    ) -> float | None:
        min_len = period + 1
        if (
            high_prices is None
            or low_prices is None
            or close_prices is None
            or len(high_prices) < min_len
            or len(low_prices) < min_len
            or len(close_prices) < min_len
        ):
            return None
        try:
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
            return float(atr) if pd.notna(atr) else None
        except Exception:
            return None

    # --- Trading Logic Methods ---

    def calculate_order_size(self, current_price: float) -> float:
        """Calculates order size in quote currency, adjusted for volatility if enabled."""
        if self.market_info is None:
            return 0.0  # Should not happen if init worked
        quote_currency = self.market_info["quote"]
        balance = self.fetch_balance(currency_code=quote_currency)
        if balance <= 0:
            logger.warning(
                f"{Fore.YELLOW}Insufficient balance ({balance:.4f} {quote_currency}) for order size calc.{Style.RESET_ALL}"
            )
            return 0.0

        base_order_size_quote = balance * self.order_size_percentage
        final_size_quote = base_order_size_quote

        # Volatility Adjustment
        if self.volatility_multiplier > 0:
            # Fetching history here adds latency, consider passing indicators if already calculated
            hist_data_vol = self.fetch_historical_data(limit=self.volatility_window + 5)
            volatility = None
            if hist_data_vol is not None and not hist_data_vol.empty:
                volatility = self.calculate_volatility(hist_data_vol["close"], self.volatility_window)
            if volatility is not None and volatility > 1e-9:
                size_factor = 1 / (1 + volatility * self.volatility_multiplier * 100)
                size_factor = max(0.25, min(1.75, size_factor))
                final_size_quote = base_order_size_quote * size_factor
                logger.info(
                    f"Volatility ({volatility:.5f}) adjusted size factor: {size_factor:.3f}. Adjusted size: {final_size_quote:.4f} {quote_currency}"
                )
            else:
                logger.debug("Volatility adjustment skipped (no data or low volatility).")
        else:
            logger.debug("Volatility adjustment disabled.")

        # Check Min Cost
        min_cost = self.market_info.get("limits", {}).get("cost", {}).get("min")
        if min_cost is not None and final_size_quote < min_cost:
            logger.warning(
                f"{Fore.YELLOW}Order size {final_size_quote:.4f} {quote_currency} below min cost {min_cost}. Setting size to 0.{Style.RESET_ALL}"
            )
            return 0.0

        # Check Min Amount
        if current_price <= 1e-9:
            return 0.0
        order_size_base = final_size_quote / current_price
        try:
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base)
            amount_precise = float(amount_precise_str)
            min_amount = self.market_info.get("limits", {}).get("amount", {}).get("min")
            if min_amount is not None and amount_precise < min_amount:
                logger.warning(
                    f"{Fore.YELLOW}Order amount {amount_precise:.{self.market_info['precision']['amount']}f} {self.market_info['base']} below min amount {min_amount}. Setting size to 0.{Style.RESET_ALL}"
                )
                return 0.0
        except Exception as e:
            logger.error(f"{Fore.RED}Error checking/applying amount precision/limits: {e}{Style.RESET_ALL}")
            return 0.0

        logger.info(
            f"{Fore.CYAN}Calculated order size: {final_size_quote:.{self.market_info['precision']['price']}f} {quote_currency} ({amount_precise:.{self.market_info['precision']['amount']}f} {self.market_info['base']}){Style.RESET_ALL}"
        )
        return final_size_quote  # Return size in QUOTE currency

    def _calculate_sl_tp_prices(
        self, entry_price: float, side: str, current_price: float, atr: float | None
    ) -> tuple[float | None, float | None]:
        """Calculates SL and TP prices based on config (ATR or fixed %)."""
        stop_loss_price = None
        take_profit_price = None
        price_precision = self.market_info["precision"]["price"]

        if self.use_atr_sl_tp:
            if atr is None or atr <= 0:
                logger.warning(
                    f"{Fore.YELLOW}ATR SL/TP enabled but ATR is invalid ({atr}). Cannot calculate SL/TP.{Style.RESET_ALL}"
                )
                return None, None
            logger.debug(
                f"Calculating SL/TP using ATR={atr:.5f}, SL Mult={self.atr_sl_multiplier}, TP Mult={self.atr_tp_multiplier}"
            )
            sl_delta = atr * self.atr_sl_multiplier
            tp_delta = atr * self.atr_tp_multiplier
            if side == "buy":
                stop_loss_price = entry_price - sl_delta
                take_profit_price = entry_price + tp_delta
            else:  # sell
                stop_loss_price = entry_price + sl_delta
                take_profit_price = entry_price - tp_delta
        else:
            if self.base_stop_loss_pct is None or self.base_take_profit_pct is None:
                logger.warning(
                    f"{Fore.YELLOW}Fixed % SL/TP enabled but percentages not set in config. Cannot calculate SL/TP.{Style.RESET_ALL}"
                )
                return None, None
            logger.debug(
                f"Calculating SL/TP using Fixed %: SL={self.base_stop_loss_pct * 100:.2f}%, TP={self.base_take_profit_pct * 100:.2f}%"
            )
            if side == "buy":
                stop_loss_price = entry_price * (1 - self.base_stop_loss_pct)
                take_profit_price = entry_price * (1 + self.base_take_profit_pct)
            else:  # sell
                stop_loss_price = entry_price * (1 + self.base_stop_loss_pct)
                take_profit_price = entry_price * (1 - self.base_take_profit_pct)

        # Apply precision
        try:
            if stop_loss_price is not None:
                stop_loss_price = float(self.exchange.price_to_precision(self.symbol, stop_loss_price))
            if take_profit_price is not None:
                take_profit_price = float(self.exchange.price_to_precision(self.symbol, take_profit_price))
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error applying precision to SL/TP prices: {e}. SL={stop_loss_price}, TP={take_profit_price}{Style.RESET_ALL}"
            )
            return None, None

        # Sanity check & adjustment
        if side == "buy" and stop_loss_price is not None and stop_loss_price >= current_price:
            adj_sl = current_price * 0.999
            logger.warning(
                f"{Fore.YELLOW}Calculated SL ({stop_loss_price:.{price_precision}f}) >= current price ({current_price:.{price_precision}f}) for LONG. Adjusting to {adj_sl:.{price_precision}f}.{Style.RESET_ALL}"
            )
            stop_loss_price = float(self.exchange.price_to_precision(self.symbol, adj_sl))
        elif side == "sell" and stop_loss_price is not None and stop_loss_price <= current_price:
            adj_sl = current_price * 1.001
            logger.warning(
                f"{Fore.YELLOW}Calculated SL ({stop_loss_price:.{price_precision}f}) <= current price ({current_price:.{price_precision}f}) for SHORT. Adjusting to {adj_sl:.{price_precision}f}.{Style.RESET_ALL}"
            )
            stop_loss_price = float(self.exchange.price_to_precision(self.symbol, adj_sl))

        logger.debug(f"Calculated SL={stop_loss_price}, TP={take_profit_price} for {side} entry at {entry_price}")
        return stop_loss_price, take_profit_price

    def compute_trade_signal_score(
        self,
        price: float,
        indicators: dict[str, float | tuple[float | None, ...] | None],
        orderbook_imbalance: float | None,
    ) -> tuple[int, list[str]]:
        """Computes a trade signal score based on indicators and order book."""
        # (Implementation remains same as V3 Bybit-mod)
        score = 0.0
        reasons = []
        RSI_OVERSOLD, RSI_OVERBOUGHT = 35, 65
        STOCH_OVERSOLD, STOCH_OVERBOUGHT = 25, 75

        # 1. Order Book Imbalance (+/- 1)
        if orderbook_imbalance is not None:
            imb_buy_thresh = 1.0 / self.imbalance_threshold
            if orderbook_imbalance < imb_buy_thresh:
                score += 1.0
                reasons.append(
                    f"{Fore.GREEN}[+1.0] OB Buy Pressure (Imb: {orderbook_imbalance:.2f} < {imb_buy_thresh:.2f}){Style.RESET_ALL}"
                )
            elif orderbook_imbalance > self.imbalance_threshold:
                score -= 1.0
                reasons.append(
                    f"{Fore.RED}[-1.0] OB Sell Pressure (Imb: {orderbook_imbalance:.2f} > {self.imbalance_threshold:.2f}){Style.RESET_ALL}"
                )
            else:
                reasons.append(f"{Fore.WHITE}[ 0.0] OB Balanced (Imb: {orderbook_imbalance:.2f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] OB Data N/A{Style.RESET_ALL}")

        # 2. EMA Trend (+/- 1)
        ema = indicators.get("ema")
        if ema is not None:
            if price > ema * 1.0002:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] Price > EMA ({price:.2f} > {ema:.2f}){Style.RESET_ALL}")
            elif price < ema * 0.9998:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] Price < EMA ({price:.2f} < {ema:.2f}){Style.RESET_ALL}")
            else:
                reasons.append(f"{Fore.WHITE}[ 0.0] Price near EMA ({price:.2f} ~ {ema:.2f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] EMA N/A{Style.RESET_ALL}")

        # 3. RSI Momentum/OB/OS (+/- 1)
        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < RSI_OVERSOLD:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] RSI Oversold ({rsi:.1f} < {RSI_OVERSOLD}){Style.RESET_ALL}")
            elif rsi > RSI_OVERBOUGHT:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] RSI Overbought ({rsi:.1f} > {RSI_OVERBOUGHT}){Style.RESET_ALL}")
            else:
                reasons.append(f"{Fore.WHITE}[ 0.0] RSI Neutral ({rsi:.1f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] RSI N/A{Style.RESET_ALL}")

        # 4. MACD Momentum/Cross (+/- 1)
        macd_line, macd_signal = indicators.get("macd_line"), indicators.get("macd_signal")
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                score += 1.0
                reasons.append(
                    f"{Fore.GREEN}[+1.0] MACD Line > Signal ({macd_line:.4f} > {macd_signal:.4f}){Style.RESET_ALL}"
                )
            else:
                score -= 1.0
                reasons.append(
                    f"{Fore.RED}[-1.0] MACD Line <= Signal ({macd_line:.4f} <= {macd_signal:.4f}){Style.RESET_ALL}"
                )
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] MACD N/A{Style.RESET_ALL}")

        # 5. Stochastic RSI OB/OS (+/- 1)
        stoch_k, stoch_d = indicators.get("stoch_k"), indicators.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if stoch_k < STOCH_OVERSOLD and stoch_d < STOCH_OVERSOLD:
                score += 1.0
                reasons.append(
                    f"{Fore.GREEN}[+1.0] StochRSI Oversold (K={stoch_k:.1f}, D={stoch_d:.1f} < {STOCH_OVERSOLD}){Style.RESET_ALL}"
                )
            elif stoch_k > STOCH_OVERBOUGHT and stoch_d > STOCH_OVERBOUGHT:
                score -= 1.0
                reasons.append(
                    f"{Fore.RED}[-1.0] StochRSI Overbought (K={stoch_k:.1f}, D={stoch_d:.1f} > {STOCH_OVERBOUGHT}){Style.RESET_ALL}"
                )
            else:
                reasons.append(
                    f"{Fore.WHITE}[ 0.0] StochRSI Neutral (K={stoch_k:.1f}, D={stoch_d:.1f}){Style.RESET_ALL}"
                )
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] StochRSI N/A{Style.RESET_ALL}")

        final_score = int(round(score))
        logger.debug(f"Signal score calculated: {final_score} (Raw: {score:.2f})")
        return final_score, reasons

    @retry_api_call(max_retries=2, initial_delay=2)
    def place_entry_order(
        self,
        side: str,
        order_size_quote: float,
        confidence_level: int,
        order_type: str,
        current_price: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict[str, Any] | None:
        """Places an entry order (market or limit), including Bybit V5 SL/TP parameters."""
        if self.market_info is None:
            return None  # Guard
        base_currency = self.market_info["base"]
        quote_currency = self.market_info["quote"]
        amount_precision = self.market_info["precision"]["amount"]
        price_precision = self.market_info["precision"]["price"]

        if current_price <= 1e-9:
            return None
        order_size_base = order_size_quote / current_price

        # --- Prepare Base Parameters and Precision ---
        params = {}
        try:
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base)
            amount_precise = float(amount_precise_str)

            limit_price_precise = None
            if order_type == "limit":
                offset = (
                    self.limit_order_entry_offset_pct_buy if side == "buy" else self.limit_order_entry_offset_pct_sell
                )
                price_factor = (1 - offset) if side == "buy" else (1 + offset)
                limit_price = current_price * price_factor
                limit_price_precise_str = self.exchange.price_to_precision(self.symbol, limit_price)
                limit_price_precise = float(limit_price_precise_str)

            # --- Add Bybit V5 SL/TP & Trigger Params ---
            if stop_loss_price is not None:
                params["stopLossPrice"] = self.exchange.price_to_precision(self.symbol, stop_loss_price)
                if self.sl_trigger_by:
                    params["slTriggerBy"] = self.sl_trigger_by  # Add trigger type if configured
            if take_profit_price is not None:
                params["takeProfitPrice"] = self.exchange.price_to_precision(self.symbol, take_profit_price)
                if self.tp_trigger_by:
                    params["tpTriggerBy"] = self.tp_trigger_by  # Add trigger type if configured

            # Check limits
            limits = self.market_info.get("limits", {})
            min_amount = limits.get("amount", {}).get("min")
            min_cost = limits.get("cost", {}).get("min")
            if min_amount is not None and amount_precise < min_amount:
                logger.warning(
                    f"{Fore.YELLOW}Entry amount {amount_precise:.{amount_precision}f} {base_currency} below min {min_amount}. Cannot place.{Style.RESET_ALL}"
                )
                return None
            cost_estimate = amount_precise * (limit_price_precise or current_price)
            if min_cost is not None and cost_estimate < min_cost:
                logger.warning(
                    f"{Fore.YELLOW}Estimated entry cost {cost_estimate:.{price_precision}f} {quote_currency} below min {min_cost}. Cannot place.{Style.RESET_ALL}"
                )
                return None

        except Exception as e:
            logger.error(f"{Fore.RED}Error preparing entry order values/params: {e}{Style.RESET_ALL}", exc_info=True)
            return None

        log_color = Fore.GREEN if side == "buy" else Fore.RED
        action_desc = f"{order_type.upper()} {side.upper()} ENTRY"
        sl_tp_info = f"SL={params.get('stopLossPrice', 'N/A')} ({params.get('slTriggerBy', 'Def')}), TP={params.get('takeProfitPrice', 'N/A')} ({params.get('tpTriggerBy', 'Def')})"

        # --- Simulation ---
        if self.simulation_mode:
            sim_id = f"sim_entry_{int(time.time() * 1000)}_{side[:1]}"
            sim_entry_price = limit_price_precise if order_type == "limit" else current_price
            sim_cost = amount_precise * sim_entry_price
            sim_status = "open" if order_type == "limit" else "closed"
            simulated_order = {
                "id": sim_id,
                "clientOrderId": sim_id,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.to_datetime("now", utc=True).isoformat(),
                "symbol": self.symbol,
                "type": order_type,
                "side": side,
                "amount": amount_precise,
                "price": limit_price_precise if order_type == "limit" else None,
                "average": sim_entry_price if sim_status == "closed" else None,
                "cost": sim_cost if sim_status == "closed" else 0.0,
                "status": sim_status,
                "filled": amount_precise if sim_status == "closed" else 0.0,
                "remaining": 0.0 if sim_status == "closed" else amount_precise,
                "info": {
                    "simulated": True,
                    "confidence": confidence_level,
                    "initial_quote_size": order_size_quote,
                    "stopLossPrice": params.get("stopLossPrice"),
                    "takeProfitPrice": params.get("takeProfitPrice"),
                    "slTriggerBy": params.get("slTriggerBy"),
                    "tpTriggerBy": params.get("tpTriggerBy"),
                },
            }
            logger.info(
                f"{log_color}[SIMULATION] Placing {action_desc}: "
                f"ID: {sim_id}, Size: {amount_precise:.{amount_precision}f} {base_currency}, "
                f"Price: {limit_price_precise or '~' + str(current_price):.{price_precision}f}, "
                f"Est. Value: {sim_cost:.2f} {quote_currency}, Confidence: {confidence_level}, {sl_tp_info}{Style.RESET_ALL}"
            )
            return simulated_order

        # --- Live Trading ---
        else:
            logger.info(f"{log_color}Attempting to place LIVE {action_desc} order with {sl_tp_info}...")
            logger.info(
                f" -> Details: Size={amount_precise:.{amount_precision}f} {base_currency}, Limit={limit_price_precise}, Params={params}"
            )
            order = None
            try:
                if order_type == "market":
                    order = self.exchange.create_market_order(self.symbol, side, amount_precise, params=params)
                elif order_type == "limit":
                    order = self.exchange.create_limit_order(
                        self.symbol, side, amount_precise, limit_price_precise, params=params
                    )

                if order:
                    oid = order.get("id", "N/A")
                    otype = order.get("type", "N/A")
                    oside = (order.get("side") or "N/A").upper()
                    oamt = order.get("amount", 0.0)
                    ofilled = order.get("filled", 0.0)
                    oprice = order.get("price") or limit_price_precise
                    oavg = order.get("average")
                    ocost = order.get("cost", 0.0)
                    ostatus = order.get("status", STATUS_UNKNOWN)
                    info_sl = order.get("info", {}).get("stopLossPrice", params.get("stopLossPrice", "N/A"))
                    info_tp = order.get("info", {}).get("takeProfitPrice", params.get("takeProfitPrice", "N/A"))
                    info_sl_trig = order.get("info", {}).get("slTriggerBy", params.get("slTriggerBy", "Def"))
                    info_tp_trig = order.get("info", {}).get("tpTriggerBy", params.get("tpTriggerBy", "Def"))

                    log_price = (
                        f"Price: {oprice:.{price_precision}f}"
                        if oprice
                        else f"Market (~{current_price:.{price_precision}f})"
                    )
                    if oavg:
                        log_price += f", Avg: {oavg:.{price_precision}f}"

                    logger.info(
                        f"{log_color}---> LIVE {action_desc} Order Placed: "
                        f"ID: {oid}, Type: {otype}, Side: {oside}, "
                        f"Amount: {oamt:.{amount_precision}f}, Filled: {ofilled:.{amount_precision}f}, {log_price}, Cost: {ocost:.2f}, "
                        f"Status: {ostatus}, SL: {info_sl} ({info_sl_trig}), TP: {info_tp} ({info_tp_trig}), Confidence: {confidence_level}{Style.RESET_ALL}"
                    )
                    order["bot_custom_info"] = {"confidence": confidence_level, "initial_quote_size": order_size_quote}
                    return order
                else:
                    logger.error(
                        f"{Fore.RED}LIVE {action_desc} order placement failed: API returned None.{Style.RESET_ALL}"
                    )
                    return None

            except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                logger.error(f"{Fore.RED}LIVE {action_desc} Order Failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
                if isinstance(e, ccxt.InvalidOrder):
                    logger.error(f" -> Params Sent: {params}")
                if isinstance(e, ccxt.InsufficientFunds):
                    self.fetch_balance(quote_currency)
                return None
            except Exception as e:
                logger.error(
                    f"{Fore.RED}LIVE {action_desc} Order Failed (Unexpected Error): {e}{Style.RESET_ALL}", exc_info=True
                )
                return None

    @retry_api_call(max_retries=1)
    def cancel_order_by_id(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancels a single order by its ID."""
        if not order_id:
            return False
        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}Attempting to cancel order {order_id} ({target_symbol})...{Style.RESET_ALL}")
        try:
            # Some exchanges might need params for cancellation (e.g., {'orderType': 'Limit'}) - check docs if needed
            self.exchange.cancel_order(order_id, target_symbol)
            logger.info(f"{Fore.YELLOW}---> Successfully cancelled order {order_id}.{Style.RESET_ALL}")
            return True
        except ccxt.OrderNotFound:
            logger.warning(
                f"{Fore.YELLOW}Order {order_id} not found for cancellation (already closed/cancelled?).{Style.RESET_ALL}"
            )
            return True  # Treat as success if already gone
        except ccxt.NetworkError as e:
            logger.error(
                f"{Fore.RED}Network error cancelling order {order_id}: {e}. Retrying if possible.{Style.RESET_ALL}"
            )
            raise e
        except ccxt.ExchangeError as e:
            # Log specific exchange errors during cancellation
            logger.error(f"{Fore.RED}Exchange error cancelling order {order_id}: {e}{Style.RESET_ALL}")
            return False
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error cancelling order {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
            return False

    @retry_api_call()
    def cancel_all_symbol_orders(self, symbol: str | None = None) -> int:
        """Cancels all open orders for the specified symbol."""
        # (Implementation remains same as V3 Bybit-mod)
        target_symbol = symbol or self.symbol
        logger.info(f"Checking and cancelling all open orders for {target_symbol}...")
        cancelled_count = 0
        try:
            if self.exchange.has["fetchOpenOrders"]:
                open_orders = self.exchange.fetch_open_orders(target_symbol)
                if not open_orders:
                    logger.info(f"No open orders found for {target_symbol}.")
                    return 0
                logger.warning(
                    f"{Fore.YELLOW}Found {len(open_orders)} open order(s) for {target_symbol}. Attempting cancellation...{Style.RESET_ALL}"
                )
                for order in open_orders:
                    if self.cancel_order_by_id(order.get("id"), target_symbol):
                        cancelled_count += 1
                    # Optional: Small delay between cancellations
                    time.sleep(max(0.1, self.exchange.rateLimit / 1000))
            elif self.exchange.has["cancelAllOrders"]:
                logger.warning(
                    f"{Fore.YELLOW}Exchange lacks fetchOpenOrders, attempting cancelAllOrders for {target_symbol}...{Style.RESET_ALL}"
                )
                response = self.exchange.cancel_all_orders(target_symbol)
                logger.info(f"cancelAllOrders response: {response}")
                cancelled_count = -1  # Indicate unknown count
            else:
                logger.error(
                    f"{Fore.RED}Exchange does not support fetchOpenOrders or cancelAllOrders. Cannot cancel automatically.{Style.RESET_ALL}"
                )
                return 0
            logger.info(
                f"Order cancellation finished for {target_symbol}. Cancelled: {cancelled_count if cancelled_count >= 0 else 'Unknown'}."
            )
            return cancelled_count
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error during bulk order cancellation for {target_symbol}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            return cancelled_count

    # --- Position Management Logic ---

    def _check_pending_entries(self, indicators: dict) -> None:
        """Checks status of pending limit entry orders and updates state if filled."""
        if not any(pos["status"] == STATUS_PENDING_ENTRY for pos in self.open_positions):
            return

        logger.debug("Checking status of pending entry orders...")
        positions_to_update = []
        current_price = self.fetch_market_price()  # Needed for SL/TP calculation context

        for position in self.open_positions:
            if position["status"] == STATUS_PENDING_ENTRY:
                entry_order_id = position["id"]
                order_info = self.fetch_order_status(entry_order_id)

                if order_info is None:
                    logger.warning(
                        f"{Fore.YELLOW}Pending entry order {entry_order_id} status unknown/not found. Removing.{Style.RESET_ALL}"
                    )
                    positions_to_update.append({"id": entry_order_id, "action": "remove"})
                    continue

                order_status = order_info.get("status")
                filled_amount = order_info.get("filled", 0.0)
                entry_price = order_info.get("average")  # Use average fill price

                if order_status == "closed":
                    if entry_price is None:
                        logger.error(
                            f"{Fore.RED}Entry order {entry_order_id} closed but average price missing! Cannot manage. Removing.{Style.RESET_ALL}"
                        )
                        positions_to_update.append({"id": entry_order_id, "action": "remove"})
                        continue

                    orig_size = position.get("original_size", position["size"])  # Use original if stored, else current
                    if (
                        filled_amount is None or abs(filled_amount - orig_size) / orig_size > 0.05
                    ):  # Check > 5% deviation
                        logger.warning(
                            f"{Fore.YELLOW}Entry order {entry_order_id} closed but filled amount ({filled_amount}) differs >5% from requested ({orig_size}). Using filled amount.{Style.RESET_ALL}"
                        )

                    logger.info(
                        f"{Fore.GREEN}Pending entry order {entry_order_id} filled! Amount: {filled_amount}, Avg Price: {entry_price}{Style.RESET_ALL}"
                    )

                    # Update position state
                    position["size"] = filled_amount  # Update size
                    position["entry_price"] = entry_price
                    position["status"] = STATUS_ACTIVE
                    position["entry_time"] = (
                        order_info.get("timestamp", time.time() * 1000) / 1000
                    )  # Use fill time if available
                    position["last_update_time"] = time.time()

                    # Calculate and STORE SL/TP prices
                    if current_price:
                        atr_value = indicators.get("atr")
                        sl_price, tp_price = self._calculate_sl_tp_prices(
                            entry_price=position["entry_price"],
                            side=position["side"],
                            current_price=current_price,
                            atr=atr_value,
                        )
                        position["stop_loss_price"] = sl_price
                        position["take_profit_price"] = tp_price
                        logger.info(f"Stored SL={sl_price}, TP={tp_price} for activated position {entry_order_id}")
                    else:
                        logger.error(
                            f"{Fore.RED}Could not fetch current price after entry fill for {entry_order_id}. SL/TP prices not calculated!{Style.RESET_ALL}"
                        )

                    positions_to_update.append({"id": entry_order_id, "action": "update", "data": position})

                elif order_status in ["canceled", "rejected", "expired"]:
                    reason = order_info.get("info", {}).get("rejectReason", "Unknown")  # Try to get rejection reason
                    logger.warning(
                        f"{Fore.YELLOW}Pending entry order {entry_order_id} failed (Status: {order_status}, Reason: {reason}). Removing.{Style.RESET_ALL}"
                    )
                    positions_to_update.append({"id": entry_order_id, "action": "remove"})

        # Apply updates
        if positions_to_update:
            new_positions_list = []
            ids_to_remove = {p["id"] for p in positions_to_update if p["action"] == "remove"}
            updates_dict = {p["id"]: p["data"] for p in positions_to_update if p["action"] == "update"}
            for pos in self.open_positions:
                if pos["id"] in ids_to_remove:
                    continue
                elif pos["id"] in updates_dict:
                    new_positions_list.append(updates_dict[pos["id"]])
                else:
                    new_positions_list.append(pos)
            self.open_positions = new_positions_list
            self._save_state()

    def _manage_active_positions(self, current_price: float) -> None:
        """Manages active positions: checks main order status, TSL, time exits."""
        if not any(pos.get("status") == STATUS_ACTIVE for pos in self.open_positions):
            return

        logger.debug(
            f"Managing active positions against current price: {current_price:.{self.market_info['precision']['price']}f}..."
        )
        positions_to_remove_ids = []
        positions_to_update_state = {}  # Store state updates (like TSL price)

        for position in self.open_positions:
            if position.get("status") != STATUS_ACTIVE:
                continue

            pos_id = position["id"]
            side = position["side"]
            entry_price = position["entry_price"]
            entry_time = position["entry_time"]
            stored_sl_price = position.get("stop_loss_price")
            stored_tp_price = position.get("take_profit_price")
            trailing_sl_price = position.get("trailing_stop_price")
            is_tsl_active = trailing_sl_price is not None

            exit_reason = None
            exit_price = None

            # --- 1. Check Status of the Main Entry Order ---
            order_info = self.fetch_order_status(pos_id)

            if order_info is None:
                logger.warning(
                    f"{Fore.YELLOW}Active position's main order {pos_id} not found. Assuming closed/cancelled externally. Removing.{Style.RESET_ALL}"
                )
                exit_reason = f"Main Order Vanished ({pos_id})"
            elif order_info.get("status") == "closed":
                exit_price = order_info.get("average") or current_price  # Use current as fallback
                stop_loss_hit_price = order_info.get("info", {}).get("stopLossPrice")  # Check if SL price is in info
                take_profit_hit_price = order_info.get("info", {}).get(
                    "takeProfitPrice"
                )  # Check if TP price is in info

                # Infer reason more reliably if possible
                if (
                    stop_loss_hit_price and abs(float(stop_loss_hit_price) - exit_price) / exit_price < 0.001
                ):  # SL triggered
                    exit_reason = f"Stop Loss Triggered (Order {pos_id})"
                    logger.info(
                        f"{Fore.RED}{exit_reason} at ~{exit_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}"
                    )
                elif (
                    take_profit_hit_price and abs(float(take_profit_hit_price) - exit_price) / exit_price < 0.001
                ):  # TP triggered
                    exit_reason = f"Take Profit Triggered (Order {pos_id})"
                    logger.info(
                        f"{Fore.GREEN}{exit_reason} at ~{exit_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}"
                    )
                else:  # Infer based on price comparison (less reliable)
                    sl_triggered = stored_sl_price and (
                        (side == "buy" and exit_price <= stored_sl_price * 1.001)
                        or (side == "sell" and exit_price >= stored_sl_price * 0.999)
                    )
                    tp_triggered = stored_tp_price and (
                        (side == "buy" and exit_price >= stored_tp_price * 0.999)
                        or (side == "sell" and exit_price <= stored_tp_price * 1.001)
                    )
                    if sl_triggered:
                        exit_reason = f"Stop Loss Triggered (Inferred - Order {pos_id})"
                    elif tp_triggered:
                        exit_reason = f"Take Profit Triggered (Inferred - Order {pos_id})"
                    else:
                        exit_reason = f"Order Closed (Reason unclear - {pos_id})"
                    log_color = Fore.RED if sl_triggered else (Fore.GREEN if tp_triggered else Fore.YELLOW)
                    logger.info(
                        f"{log_color}{exit_reason} at ~{exit_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}"
                    )

            # --- 2. Time-Based Exit Check (if order still open) ---
            elif not exit_reason and self.time_based_exit_minutes and self.time_based_exit_minutes > 0:
                time_elapsed_minutes = (time.time() - entry_time) / 60
                if time_elapsed_minutes >= self.time_based_exit_minutes:
                    logger.info(
                        f"{Fore.YELLOW}Time limit ({self.time_based_exit_minutes} min) reached for position {pos_id}. Initiating market close.{Style.RESET_ALL}"
                    )
                    exit_reason = "Time Limit Reached"
                    if self.cancel_order_by_id(pos_id):  # Cancel original order first
                        market_close_order = self._place_market_close_order(position, current_price)
                        if market_close_order:
                            exit_price = market_close_order.get("average", current_price)
                        else:
                            exit_reason = None
                            logger.critical(
                                f"{Fore.RED}FAILED MARKET CLOSE FOR TIME EXIT - {pos_id} REMAINS OPEN! MANUAL INTERVENTION!{Style.RESET_ALL}"
                            )
                    else:
                        exit_reason = None
                        logger.error(
                            f"{Fore.RED}Failed to cancel original order {pos_id} for time exit. Close aborted.{Style.RESET_ALL}"
                        )

            # --- 3. Trailing Stop Logic (if enabled and order still open) ---
            # *** WARNING: TSL using edit_order needs careful verification with your CCXT version and Bybit V5 ***
            if not exit_reason and self.enable_trailing_stop_loss:
                new_tsl_price = None
                # Activation
                if not is_tsl_active:
                    activate_threshold = None
                    if stored_tp_price:  # Activate halfway to TP
                        activate_threshold = (
                            entry_price + (stored_tp_price - entry_price) * 0.5
                            if side == "buy"
                            else entry_price - (entry_price - stored_tp_price) * 0.5
                        )
                    if activate_threshold and (
                        (side == "buy" and current_price >= activate_threshold)
                        or (side == "sell" and current_price <= activate_threshold)
                    ):
                        tsl_factor = (
                            (1 - self.trailing_stop_loss_percentage)
                            if side == "buy"
                            else (1 + self.trailing_stop_loss_percentage)
                        )
                        potential_tsl = current_price * tsl_factor
                        breakeven_plus = entry_price * (1 + 0.0005) if side == "buy" else entry_price * (1 - 0.0005)
                        new_tsl_price = (
                            max(potential_tsl, breakeven_plus) if side == "buy" else min(potential_tsl, breakeven_plus)
                        )
                        new_tsl_price = float(self.exchange.price_to_precision(self.symbol, new_tsl_price))
                        logger.info(
                            f"{Fore.MAGENTA}Trailing Stop ACTIVATED for {side.upper()} {pos_id} at {new_tsl_price:.{self.market_info['precision']['price']}f} (Price: {current_price:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}"
                        )
                # Update
                elif is_tsl_active:
                    tsl_factor = (
                        (1 - self.trailing_stop_loss_percentage)
                        if side == "buy"
                        else (1 + self.trailing_stop_loss_percentage)
                    )
                    potential_tsl = current_price * tsl_factor
                    if (side == "buy" and potential_tsl > trailing_sl_price) or (
                        side == "sell" and potential_tsl < trailing_sl_price
                    ):
                        new_tsl_price = float(self.exchange.price_to_precision(self.symbol, potential_tsl))
                        logger.info(
                            f"{Fore.MAGENTA}Trailing Stop UPDATED for {side.upper()} {pos_id} to {new_tsl_price:.{self.market_info['precision']['price']}f} (Price: {current_price:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}"
                        )

                # Apply TSL Update via edit_order (NEEDS VERIFICATION)
                if new_tsl_price is not None:
                    logger.warning(
                        f"{Fore.YELLOW}Attempting TSL update via edit_order for {pos_id} to SL={new_tsl_price}. VERIFY EXCHANGE/CCXT SUPPORT!{Style.RESET_ALL}"
                    )
                    try:
                        edit_params = {"stopLossPrice": self.exchange.price_to_precision(self.symbol, new_tsl_price)}
                        if self.sl_trigger_by:
                            edit_params["slTriggerBy"] = self.sl_trigger_by  # Include trigger type if set
                        # Pass other required params for edit_order (might need type, side, amount etc.) - CHECK CCXT DOCS
                        edited_order = self.exchange.edit_order(
                            id=pos_id,
                            symbol=self.symbol,
                            type=order_info.get("type"),  # Pass original type
                            side=order_info.get("side"),  # Pass original side
                            amount=order_info.get("amount"),  # Pass original amount
                            price=order_info.get("price"),  # Pass original price (for limit)
                            params=edit_params,
                        )
                        if edited_order:
                            logger.info(
                                f"{Fore.MAGENTA}Successfully modified order {pos_id} with new TSL price {new_tsl_price:.{self.market_info['precision']['price']}f}.{Style.RESET_ALL}"
                            )
                            positions_to_update_state[pos_id] = {
                                "trailing_stop_price": new_tsl_price,
                                "stop_loss_price": new_tsl_price,  # Update stored SL
                                "last_update_time": time.time(),
                            }
                        else:
                            logger.error(
                                f"{Fore.RED}Failed TSL update for {pos_id} (edit_order returned None).{Style.RESET_ALL}"
                            )
                    except ccxt.NotSupported as e:
                        logger.error(
                            f"{Fore.RED}TSL update failed: edit_order for SL not supported by Exchange/CCXT for {pos_id}: {e}. Disable TSL or implement cancel/replace.{Style.RESET_ALL}"
                        )
                        # Disable TSL for future iterations if not supported?
                        # self.enable_trailing_stop_loss = False
                    except Exception as e:
                        logger.error(
                            f"{Fore.RED}Error modifying order {pos_id} for TSL: {e}{Style.RESET_ALL}", exc_info=True
                        )

            # --- 4. Process Exit ---
            if exit_reason:
                positions_to_remove_ids.append(pos_id)
                # Log PnL
                if exit_price is not None:
                    pnl_quote = (
                        (exit_price - entry_price) * position["size"]
                        if side == "buy"
                        else (entry_price - exit_price) * position["size"]
                    )
                    pnl_pct = (
                        (exit_price / entry_price - 1) * 100
                        if side == "buy" and entry_price
                        else (entry_price / exit_price - 1) * 100
                        if side == "sell" and exit_price
                        else 0.0
                    )
                    pnl_color = Fore.GREEN if pnl_quote >= 0 else Fore.RED
                    logger.info(
                        f"{pnl_color}---> Position {pos_id} Closed. Reason: {exit_reason}. Est. PnL: {pnl_quote:.4f} {self.market_info['quote']} ({pnl_pct:.3f}%){Style.RESET_ALL}"
                    )
                    self.daily_pnl += pnl_quote
                else:
                    logger.warning(
                        f"{Fore.YELLOW}---> Position {pos_id} Closed. Reason: {exit_reason}. Exit price unknown, cannot calculate PnL.{Style.RESET_ALL}"
                    )

        # --- Apply Position Updates and Removals ---
        if positions_to_remove_ids or positions_to_update_state:
            new_positions_list = []
            for pos in self.open_positions:
                pos_id = pos["id"]
                if pos_id in positions_to_remove_ids:
                    continue
                if pos_id in positions_to_update_state:
                    pos.update(positions_to_update_state[pos_id])
                new_positions_list.append(pos)
            self.open_positions = new_positions_list
            self._save_state()

    def _place_market_close_order(self, position: dict[str, Any], current_price: float) -> dict[str, Any] | None:
        """Places a market order to close a given position."""
        # (Implementation remains same as V3 Bybit-mod)
        pos_id = position["id"]
        side = position["side"]
        size = position["size"]
        close_side = "sell" if side == "buy" else "buy"
        log_color = Fore.YELLOW
        logger.warning(
            f"{log_color}Placing MARKET CLOSE order for position {pos_id} (Side: {close_side}, Size: {size})...{Style.RESET_ALL}"
        )

        # Simulation
        if self.simulation_mode:
            sim_id = f"sim_close_{int(time.time() * 1000)}_{close_side[:1]}"
            size * current_price
            simulated_order = {
                "id": sim_id,
                "type": "market",
                "side": close_side,
                "amount": size,
                "average": current_price,
                "status": "closed",
                "filled": size,
                "info": {"simulated": True, "position_id": pos_id, "reduceOnly": True},
            }  # Added reduceOnly flag
            logger.info(
                f"{log_color}[SIMULATION] Market Close Order: ID {sim_id}, Size {size}, AvgPrice {current_price}{Style.RESET_ALL}"
            )
            return simulated_order

        # Live Trading
        else:
            order = None
            try:
                params = {"reduce_only": True}
                order = self.exchange.create_market_order(self.symbol, close_side, size, params=params)
                if order:
                    oid = order.get("id", "N/A")
                    oavg = order.get("average", current_price)
                    ostatus = order.get("status", STATUS_UNKNOWN)
                    logger.info(
                        f"{log_color}---> LIVE Market Close Order Placed: ID {oid}, AvgPrice ~{oavg:.{self.market_info['precision']['price']}f}, Status {ostatus}{Style.RESET_ALL}"
                    )
                    return order
                else:
                    logger.error(
                        f"{Fore.RED}LIVE Market Close order placement failed: API returned None.{Style.RESET_ALL}"
                    )
                    return None
            except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                logger.error(f"{Fore.RED}LIVE Market Close Order Failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
                return None
            except Exception as e:
                logger.error(
                    f"{Fore.RED}LIVE Market Close Order Failed (Unexpected Error): {e}{Style.RESET_ALL}", exc_info=True
                )
                return None

    # --- Main Loop Structure ---

    def _fetch_market_data(self) -> dict[str, Any] | None:
        """Fetches all required market data for an iteration."""
        logger.debug("Fetching market data bundle...")
        current_price = self.fetch_market_price()
        order_book_data = self.fetch_order_book()
        historical_data = self.fetch_historical_data()

        if current_price is None or historical_data is None or historical_data.empty or order_book_data is None:
            logger.warning(f"{Fore.YELLOW}Incomplete market data fetched. Skipping iteration.{Style.RESET_ALL}")
            return None

        return {
            "price": current_price,
            "order_book_imbalance": order_book_data.get("imbalance"),
            "historical_data": historical_data,
        }

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> dict[str, Any]:
        """Calculates all technical indicators."""
        logger.debug("Calculating indicators...")
        indicators = {}
        close = historical_data["close"]
        high = historical_data["high"]
        low = historical_data["low"]

        indicators["volatility"] = self.calculate_volatility(close, self.volatility_window)
        indicators["ema"] = self.calculate_ema(close, self.ema_period)
        indicators["rsi"] = self.calculate_rsi(close, self.rsi_period)
        indicators["macd_line"], indicators["macd_signal"], indicators["macd_hist"] = self.calculate_macd(
            close, self.macd_short_period, self.macd_long_period, self.macd_signal_period
        )
        indicators["stoch_k"], indicators["stoch_d"] = self.calculate_stoch_rsi(
            close, self.rsi_period, self.stoch_rsi_period, self.stoch_rsi_k_period, self.stoch_rsi_d_period
        )
        indicators["atr"] = self.calculate_atr(high, low, close, self.atr_period)

        # Log indicators
        log_inds = {
            k: f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else v) for k, v in indicators.items()
        }
        self.market_info["precision"]["price"]
        logger.info(
            f"Indicators: EMA={log_inds['ema']}, RSI={log_inds['rsi']}, ATR={log_inds['atr']}, "
            f"MACD={log_inds['macd_line']}, StochK={log_inds['stoch_k']}"
        )
        return indicators

    def _process_signals_and_entry(self, market_data: dict, indicators: dict) -> None:
        """Computes signal score and attempts position entry if conditions met."""
        current_price = market_data["price"]
        orderbook_imbalance = market_data["order_book_imbalance"]
        atr_value = indicators.get("atr")  # Get ATR for SL/TP calc

        # Calculate potential order size
        order_size_quote = self.calculate_order_size(current_price)
        can_trade = order_size_quote > 0

        # Compute signal score
        signal_score, reasons = self.compute_trade_signal_score(current_price, indicators, orderbook_imbalance)
        logger.info(f"Trade Signal Score: {signal_score}")
        if abs(signal_score) >= 1 or logger.isEnabledFor(logging.DEBUG):
            for reason in reasons:
                logger.info(f" -> {reason}")

        # Check if new position can be opened
        can_open_new = len(self.open_positions) < self.max_open_positions
        if not can_open_new:
            logger.info(
                f"{Fore.YELLOW}Max open positions ({self.max_open_positions}) reached. No new entries.{Style.RESET_ALL}"
            )
            return
        if not can_trade:
            logger.warning(f"{Fore.YELLOW}Cannot enter trade: Calculated order size is zero.{Style.RESET_ALL}")
            return

        # --- Entry Logic ---
        entry_side = None
        if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = "buy"
        elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = "sell"

        if entry_side:
            log_color = Fore.GREEN if entry_side == "buy" else Fore.RED
            logger.info(
                f"{log_color}Potential {entry_side.upper()} entry signal detected (Score: {signal_score}){Style.RESET_ALL}"
            )

            # --- Calculate SL/TP Prices BEFORE placing order ---
            # Use current_price as the reference for initial SL/TP calculation
            sl_price, tp_price = self._calculate_sl_tp_prices(
                entry_price=current_price,  # Use current price as estimate for calculation
                side=entry_side,
                current_price=current_price,
                atr=atr_value,
            )
            if sl_price is None and tp_price is None:
                logger.error(f"{Fore.RED}Failed to calculate SL/TP prices. Cannot place entry order.{Style.RESET_ALL}")
                return

            # --- Place Entry Order with SL/TP Params ---
            entry_order = self.place_entry_order(
                side=entry_side,
                order_size_quote=order_size_quote,
                confidence_level=signal_score,
                order_type=self.entry_order_type,
                current_price=current_price,
                stop_loss_price=sl_price,  # Pass calculated SL
                take_profit_price=tp_price,  # Pass calculated TP
            )

            if entry_order:
                order_status = entry_order.get("status")
                order_id = entry_order["id"]
                entry_order.get("filled", 0.0)
                # Use average fill price if available (especially for market), else use limit price or current price
                actual_entry_price = entry_order.get("average") or entry_order.get("price") or current_price

                # --- Add to Bot State ---
                new_position = {
                    "id": order_id,
                    "side": entry_side,
                    "size": entry_order.get("amount"),  # Requested amount
                    "original_size": entry_order.get("amount"),  # Store requested amount
                    "entry_price": actual_entry_price,  # Best guess entry price
                    "entry_time": time.time(),
                    "status": STATUS_PENDING_ENTRY if order_status == "open" else STATUS_ACTIVE,
                    "entry_order_type": self.entry_order_type,
                    # Store the calculated SL/TP prices that were *sent* with the order
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "confidence": signal_score,
                    "trailing_stop_price": None,  # Initial TSL price is None
                    "last_update_time": time.time(),
                }
                self.open_positions.append(new_position)
                logger.info(f"Position {order_id} added to state with status: {new_position['status']}")

                # If entry was market and filled immediately, update entry price/time more accurately
                if new_position["status"] == STATUS_ACTIVE:
                    new_position["entry_price"] = entry_order.get("average", actual_entry_price)  # Prefer average
                    new_position["size"] = entry_order.get("filled", new_position["size"])  # Update to filled size
                    new_position["entry_time"] = entry_order.get("timestamp", time.time() * 1000) / 1000

                self._save_state()
        else:
            logger.info(f"Neutral signal score ({signal_score}). No entry action.")

    def run(self) -> None:
        """Starts the main trading loop."""
        logger.info(f"{Fore.CYAN}--- Initiating Trading Loop (Symbol: {self.symbol}) ---{Style.RESET_ALL}")
        # ... (Initial cleanup logic can remain) ...

        while True:
            self.iteration += 1
            start_time = time.time()
            loop_prefix = f"{Fore.BLUE}===== Iteration {self.iteration} ===={Style.RESET_ALL}"
            logger.info(f"\n{loop_prefix} Time: {pd.Timestamp.now(tz='UTC').isoformat()}")

            try:
                # --- 1. Fetch Data ---
                market_data = self._fetch_market_data()
                if market_data is None:
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue
                current_price = market_data["price"]

                # --- 2. Calculate Indicators ---
                indicators = self._calculate_indicators(market_data["historical_data"])
                logger.info(
                    f"{loop_prefix} Current Price: {current_price:.{self.market_info['precision']['price']}f}, OB Imbalance: {f'{market_data["order_book_imbalance"]:.3f}' if market_data['order_book_imbalance'] is not None else 'N/A'}"
                )

                # --- 3. Check Pending Entries ---
                self._check_pending_entries(indicators)

                # --- 4. Process Signals & Potential Entry ---
                self._process_signals_and_entry(market_data, indicators)

                # --- 5. Manage Active Positions ---
                self._manage_active_positions(current_price)

                # --- 6. Wait ---
                end_time = time.time()
                execution_time = end_time - start_time
                wait_time = max(0.1, DEFAULT_SLEEP_INTERVAL_SECONDS - execution_time)
                logger.debug(f"{loop_prefix} Loop took {execution_time:.2f}s. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.critical(
                    f"{Fore.RED}{loop_prefix} Critical error in main loop: {e}{Style.RESET_ALL}", exc_info=True
                )
                logger.warning(f"{Fore.YELLOW}Pausing for 60 seconds due to critical error...{Style.RESET_ALL}")
                time.sleep(60)

        # logger.info("Main trading loop concluded.") # Unreachable in normal operation

    def shutdown(self) -> None:
        """Performs graceful shutdown procedures."""
        logger.warning(f"{Fore.YELLOW}--- Initiating Graceful Shutdown Sequence ---{Style.RESET_ALL}")

        # 1. Cancel All Open Orders (associated with known positions)
        logger.info("Attempting to cancel orders linked to known positions...")
        cancelled_count = 0
        # Iterate over a copy in case cancellation modifies the list indirectly (shouldn't happen here)
        for pos in list(self.open_positions):
            # Only cancel if the order is likely still open (pending entry or active)
            if pos.get("status") in [STATUS_PENDING_ENTRY, STATUS_ACTIVE]:
                if self.cancel_order_by_id(pos.get("id")):  # Cancel main order (entry or active with SL/TP)
                    cancelled_count += 1
        logger.info(f"Cancelled {cancelled_count} orders linked to known positions.")
        # Optional: Attempt broader cancellation if needed
        # logger.info("Attempting to cancel any other open orders for the symbol...")
        # self.cancel_all_symbol_orders()

        # 2. Optionally Close Open Positions (Market Order)
        CLOSE_POSITIONS_ON_EXIT = False  # Default to False for safety
        active_positions = [p for p in self.open_positions if p.get("status") == STATUS_ACTIVE]
        if CLOSE_POSITIONS_ON_EXIT and active_positions:
            logger.warning(
                f"{Fore.YELLOW}Attempting market close for {len(active_positions)} active position(s)...{Style.RESET_ALL}"
            )
            current_price = self.fetch_market_price()
            if current_price:
                for position in list(active_positions):  # Iterate copy
                    close_order = self._place_market_close_order(position, current_price)
                    if close_order:
                        logger.info(
                            f"{Fore.YELLOW}Market close order placed for position {position['id']}.{Style.RESET_ALL}"
                        )
                        # Find and remove the position from the main list
                        self.open_positions = [p for p in self.open_positions if p["id"] != position["id"]]
                    else:
                        logger.error(
                            f"{Fore.RED}Failed market close for {position['id']}. Manual check needed!{Style.RESET_ALL}"
                        )
            else:
                logger.error(
                    f"{Fore.RED}Cannot fetch price for market close. Positions remain open! Manual check needed!{Style.RESET_ALL}"
                )
        elif active_positions:
            logger.warning(
                f"{Fore.YELLOW}{len(active_positions)} position(s) remain active. Manual management required.{Style.RESET_ALL}"
            )
            for pos in active_positions:
                logger.warning(f" -> ID={pos['id']}, Side={pos['side']}, Size={pos['size']}")

        # 3. Final State Save
        logger.info("Saving final state...")
        self._save_state()

        logger.info(f"{Fore.CYAN}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")
        logging.shutdown()


# --- Main Execution Block ---
if __name__ == "__main__":
    bot_instance = None
    exit_code = 0
    try:
        # Ensure state directory exists if needed
        state_dir = os.path.dirname(STATE_FILE_NAME)
        if state_dir and not os.path.exists(state_dir):
            logger.info(f"Creating state directory: {state_dir}")
            os.makedirs(state_dir)

        bot_instance = ScalpingBot(config_file=CONFIG_FILE_NAME, state_file=STATE_FILE_NAME)
        bot_instance.run()

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}Shutdown signal received (Ctrl+C).{Style.RESET_ALL}")

    except SystemExit as e:
        # Log SystemExit if it wasn't from a clean exit (code 0)
        if e.code != 0:
            logger.error(f"Bot exited via SystemExit (Code: {e.code}).")
        else:
            logger.info("Bot exited cleanly.")
        exit_code = e.code

    except Exception as e:
        logger.critical(
            f"{Fore.RED}An unhandled critical error broke the main spell: {e}{Style.RESET_ALL}", exc_info=True
        )
        exit_code = 1

    finally:
        if bot_instance:
            bot_instance.shutdown()
        else:
            logger.info("Bot instance not fully initialized. Shutting down logging.")
            logging.shutdown()
        sys.exit(exit_code)
