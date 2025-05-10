#!/usr/bin/env python3
"""Bybit Futures Terminal - v2.5 - Enhanced Edition (Standard Asyncio)
Author: Mentallyspammed1 (Enhanced by AI)
Last Updated: 2025-03-28.

Description:
A command-line interface for interacting with Bybit Futures (USDT Perpetual).
This enhanced version focuses on:
- Unified API interaction via the CCXT library for consistency and robustness.
- Improved error handling and user feedback.
- Asynchronous operations using standard asyncio.
- Configuration management via JSON and .env files.
- Basic technical analysis using pandas-ta.
- Clear, color-coded terminal UI using Colorama.
- Basic ASCII charting for price trends.
"""

import asyncio
import contextlib
import json
import logging
import os
import signal  # For graceful shutdown
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asciichartpy as asciichart

# Async & CCXT
import ccxt.async_support as ccxt
import numpy as np

# Data & Analysis
import pandas as pd
import pandas_ta as ta

# UI & Utilities
from colorama import Fore, Style, init

# Configuration & Environment
from dotenv import load_dotenv

# Configure logging
log_file = "terminal.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),  # Log to stdout as well
    ],
)
logger = logging.getLogger("BybitTerminal")

# Initialize colorama for cross-platform terminal colors
init(autoreset=True)

# --- Configuration & Credentials ---


@dataclass
class APICredentials:
    """Dataclass to hold API credentials."""

    api_key: str
    api_secret: str
    testnet: bool = False


class ConfigManager:
    """Manages the terminal configuration from config.json."""

    DEFAULT_CONFIG_PATH = "config.json"
    DEFAULT_CONFIG = {
        "theme": "dark",
        "log_level": "INFO",
        "default_symbol": "BTC/USDT",  # Use CCXT format
        "default_timeframe": "1h",
        "default_order_type": "Limit",  # Added default order type
        "connection_timeout_ms": 30000,  # CCXT uses milliseconds
        "order_history_limit": 50,
        "indicator_periods": {
            "sma": 20,
            "ema": 20,
            "rsi": 14,
            "bbands": 20,
            "bbands_std": 2.0,
        },
        "chart_height": 10,  # Height for ASCII chart
        "chart_points": 50,  # Number of data points for ASCII chart
    }

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config: dict = self._load_config()
        self._apply_log_level()
        self.theme_colors: dict = self._setup_theme_colors()
        logger.info(f"Configuration loaded from '{self.config_path}'")

    def _load_config(self) -> dict:
        """Loads configuration from JSON file or creates default."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    loaded_config = json.load(f)
                    # Merge with default to ensure all keys exist
                    merged_config = {**self.DEFAULT_CONFIG, **loaded_config}
                    # Ensure nested dicts like indicator_periods are also merged correctly
                    if "indicator_periods" in loaded_config:
                        merged_config["indicator_periods"] = {
                            **self.DEFAULT_CONFIG["indicator_periods"],
                            **loaded_config["indicator_periods"],
                        }
                    return merged_config
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON from '{self.config_path}'. Using default config."
                )
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                logger.error(f"Failed to load config: {e}. Using default config.")
                return self.DEFAULT_CONFIG.copy()
        else:
            logger.warning(
                f"'{self.config_path}' not found. Creating default configuration."
            )
            return self._create_default_config()

    def _create_default_config(self) -> dict:
        """Creates a default configuration file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=4)
            logger.info(f"Default configuration file created at '{self.config_path}'")
            return self.DEFAULT_CONFIG.copy()
        except OSError as e:
            logger.error(
                f"Error creating default config file: {e}. Using in-memory default."
            )
            return self.DEFAULT_CONFIG.copy()

    def save_config(self) -> None:
        """Saves the current configuration to the file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to '{self.config_path}'")
        except OSError as e:
            logger.error(f"Error saving configuration: {e}")

    def _apply_log_level(self) -> None:
        """Applies the log level from the configuration."""
        log_level_str = self.config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level)  # Set root logger level
        logger.setLevel(log_level)  # Set our specific logger level
        # Also update handlers if they have different levels
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level)
        logger.info(f"Log level set to {log_level_str}")

    def _setup_theme_colors(self) -> dict:
        """Sets up theme colors based on the configuration."""
        theme = self.config.get("theme", "dark")
        if theme == "dark":
            return {
                "primary": Fore.CYAN,
                "secondary": Fore.MAGENTA,
                "accent": Fore.YELLOW,
                "error": Fore.RED,
                "success": Fore.GREEN,
                "warning": Fore.YELLOW,
                "info": Fore.BLUE,
                "title": Fore.CYAN + Style.BRIGHT,
                "menu_option": Fore.WHITE,
                "menu_highlight": Fore.YELLOW + Style.BRIGHT,
                "input_prompt": Fore.YELLOW + Style.BRIGHT,
                "table_header": Fore.CYAN + Style.BRIGHT,
                "positive": Fore.GREEN,
                "negative": Fore.RED,
                "neutral": Fore.WHITE,
                "reset": Style.RESET_ALL,
            }
        else:  # Light theme (adjust colors as needed)
            return {
                "primary": Fore.BLUE,
                "secondary": Fore.GREEN,
                "accent": Fore.MAGENTA,
                "error": Fore.RED,
                "success": Fore.GREEN,
                "warning": Fore.YELLOW,
                "info": Fore.CYAN,
                "title": Fore.BLUE + Style.BRIGHT,
                "menu_option": Fore.BLACK,
                "menu_highlight": Fore.BLUE + Style.BRIGHT,
                "input_prompt": Fore.MAGENTA + Style.BRIGHT,
                "table_header": Fore.BLUE + Style.BRIGHT,
                "positive": Fore.GREEN,
                "negative": Fore.RED,
                "neutral": Fore.BLACK,
                "reset": Style.RESET_ALL,
            }


# --- CCXT Exchange Client ---


class BybitFuturesCCXTClient:
    """Client for interacting with Bybit Futures via CCXT.
    Handles initialization, context management, and core API calls.
    """

    def __init__(self, credentials: APICredentials, config: dict) -> None:
        self.credentials = credentials
        self.config = config
        self.exchange: ccxt.bybit | None = None
        self._initialized = False

    async def __aenter__(self):
        """Initializes the CCXT exchange instance asynchronously."""
        if not self._initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the CCXT exchange connection."""
        await self.close()

    async def initialize(self) -> None:
        """Sets up the CCXT Bybit exchange instance."""
        if self._initialized:
            logger.debug("CCXT client already initialized.")
            return

        logger.info("Initializing CCXT Bybit client...")
        exchange_config = {
            "apiKey": self.credentials.api_key,
            "secret": self.credentials.api_secret,
            "enableRateLimit": True,  # Use CCXT's built-in rate limiter
            "options": {
                "defaultType": "swap",  # Crucial for USDT perpetual futures
                "adjustForTimeDifference": True,  # Handle minor clock skew
                "createOrderRequiresPrice": False,  # Allow market orders without explicit price=None
            },
            "timeout": self.config.get("connection_timeout_ms", 30000),
        }
        if self.credentials.testnet:
            logger.warning("Using Bybit TESTNET environment.")
            # CCXT handles testnet via sandboxMode or specific URLs if needed
            # For Bybit, setting sandboxMode=True is standard
            exchange_config["options"]["sandboxMode"] = (
                True  # Or set testnet URLs if CCXT requires
            )

        try:
            self.exchange = ccxt.bybit(exchange_config)
            # Test connection by loading markets (essential for many operations)
            logger.info("Loading markets...")
            await self.exchange.load_markets()
            self._initialized = True
            logger.info(
                f"{Fore.GREEN}CCXT Bybit Futures client initialized successfully for {'Testnet' if self.credentials.testnet else 'Mainnet'}.{Style.RESET_ALL}"
            )
        except ccxt.AuthenticationError as e:
            logger.error(
                f"{Fore.RED}CCXT Authentication Error: Invalid API keys or permissions. {e}{Style.RESET_ALL}"
            )
            self.exchange = None  # Ensure exchange is None on failure
            raise ConnectionError("CCXT Authentication Failed") from e
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            logger.error(
                f"{Fore.RED}CCXT Network/Availability Error: Could not connect to Bybit. {e}{Style.RESET_ALL}"
            )
            self.exchange = None
            raise ConnectionError("CCXT Network Failed") from e
        except ccxt.ExchangeError as e:
            logger.error(
                f"{Fore.RED}CCXT Exchange Error during initialization: {e}{Style.RESET_ALL}"
            )
            self.exchange = None
            raise ConnectionError("CCXT Exchange Initialization Failed") from e
        except Exception as e:
            logger.error(
                f"{Fore.RED}An unexpected error occurred initializing CCXT: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            self.exchange = None
            raise ConnectionError("Unexpected CCXT Initialization Error") from e

    async def close(self) -> None:
        """Closes the underlying CCXT exchange connection."""
        if self.exchange and self._initialized:
            logger.info("Closing CCXT Bybit client connection...")
            try:
                await self.exchange.close()
                self._initialized = False
                logger.info("CCXT connection closed.")
            except Exception as e:
                logger.error(f"Error closing CCXT connection: {e}", exc_info=True)
        self.exchange = None  # Ensure it's cleared

    def _check_initialized(self) -> None:
        """Raises an error if the client is not initialized."""
        if not self.exchange or not self._initialized:
            raise ConnectionError(
                "CCXT client is not initialized or connection failed."
            )

    async def fetch_balance(self) -> dict:
        """Fetches account balance (USDT Futures)."""
        self._check_initialized()
        logger.debug("Fetching balance...")
        try:
            # Fetch balance specifically for perpetual swap markets (linear USDT)
            # CCXT standardizes this, but params might fine-tune for specific account types if needed
            balance = await self.exchange.fetch_balance(params={"type": "swap"})
            logger.debug("Raw balance fetched.")
            # Return the standardized USDT balance part
            return balance.get("USDT", {})
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT Error fetching balance: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching balance: {e}", exc_info=True)
            raise

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        params: dict = None,
    ) -> dict:
        """Places a market or limit order using CCXT."""
        if params is None:
            params = {}
        self._check_initialized()
        logger.info(
            f"Placing {order_type} {side} order: {amount} {symbol} @ {price if price else 'Market'}"
        )
        try:
            # Explicitly set reduceOnly if needed via params
            # Example: params = {'reduceOnly': True}
            # Ensure timeInForce is set if needed (e.g., GTC, IOC, FOK) via params
            # Example: params = {'timeInForce': 'IOC'}

            # Check if market needs price=None explicitly (CCXT often handles this)
            # if order_type.lower() == 'market' and 'createOrderRequiresPrice' in self.exchange.options and self.exchange.options['createOrderRequiresPrice']:
            #    price = None # Explicitly set price to None if exchange requires it for market orders

            order = await self.exchange.create_order(
                symbol, order_type, side, amount, price, params
            )

            # CCXT standardizes the order structure returned by create_order
            logger.info(f"Order placement successful: ID {order.get('id')}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(
                f"{Fore.RED}Order placement failed: Insufficient funds. {e}{Style.RESET_ALL}"
            )
            raise
        except ccxt.InvalidOrder as e:
            logger.error(
                f"{Fore.RED}Order placement failed: Invalid order parameters (size, price, symbol, type, etc.). {e}{Style.RESET_ALL}"
            )
            raise
        except ccxt.ExchangeError as e:
            logger.error(
                f"{Fore.RED}CCXT Exchange Error placing order: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error placing order: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            raise

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[list[int | float]]:
        """Fetches OHLCV data."""
        self._check_initialized()
        logger.debug(f"Fetching {limit} OHLCV candles for {symbol} ({timeframe})...")
        try:
            # Add params if needed, e.g., {'paginate': True} for some exchanges if limit is very large
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit
            )
            if not ohlcv:
                logger.warning(f"No OHLCV data returned for {symbol} ({timeframe}).")
                return []
            # CCXT returns list of lists: [timestamp, open, high, low, close, volume]
            return ohlcv
        except ccxt.BadSymbol as e:
            logger.error(
                f"{Fore.RED}Invalid symbol for OHLCV: {symbol}. {e}{Style.RESET_ALL}"
            )
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.ExchangeError as e:
            # Check if the error message specifically mentions timeframe
            if "timeframe" in str(e).lower() or "interval" in str(e).lower():
                logger.error(
                    f"{Fore.RED}Invalid or unsupported timeframe '{timeframe}' for {symbol}. {e}{Style.RESET_ALL}"
                )
                raise ValueError(f"Invalid timeframe: {timeframe}") from e
            else:
                logger.error(
                    f"{Fore.RED}CCXT Exchange Error fetching OHLCV: {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                raise
        except Exception as e:
            logger.error(
                f"{Fore.RED}Unexpected error fetching OHLCV: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            raise


# --- Technical Analysis ---


class TechnicalAnalysis:
    """Performs technical analysis on market data."""

    def __init__(self, config: dict) -> None:
        self.config = config.get("indicator_periods", {})

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates technical indicators using pandas_ta based on config."""
        if df.empty:
            logger.warning("Cannot calculate indicators on empty DataFrame.")
            return df

        # Ensure standard column names (case-insensitive check)
        df.columns = [
            col.capitalize() for col in df.columns
        ]  # Standardize to Capitalized
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"DataFrame missing required columns for TA: {missing}")
            raise ValueError(f"DataFrame missing required columns: {missing}")

        logger.debug("Calculating technical indicators...")
        df_out = df.copy()

        # Safely get lengths from config, providing defaults
        sma_len = self.config.get("sma", 20)
        ema_len = self.config.get("ema", 20)
        rsi_len = self.config.get("rsi", 14)
        bb_len = self.config.get("bbands", 20)
        bb_std = self.config.get("bbands_std", 2.0)
        # MACD defaults in pandas-ta are usually (fast=12, slow=26, signal=9)
        # We don't need to configure them unless overriding defaults

        try:
            # Apply indicators using pandas_ta strategy (optional but can simplify)
            # Or apply one by one
            custom_strategy = ta.Strategy(
                name="TA_Set",
                ta=[
                    {"kind": "sma", "length": sma_len},
                    {"kind": "ema", "length": ema_len},
                    {"kind": "rsi", "length": rsi_len},
                    {"kind": "macd"},  # Uses default lengths (12, 26, 9)
                    {"kind": "bbands", "length": bb_len, "std": bb_std},
                ],
            )
            df_out.ta.strategy(custom_strategy)

            # Check if columns were added (pandas_ta might fail silently sometimes)
            expected_cols = [
                f"SMA_{sma_len}",
                f"EMA_{ema_len}",
                f"RSI_{rsi_len}",
                "MACD_12_26_9",
                f"BBL_{bb_len}_{bb_std}",
                f"BBM_{bb_len}_{bb_std}",
                f"BBU_{bb_len}_{bb_std}",
            ]  # Example expected names
            added_cols = [col for col in expected_cols if col in df_out.columns]
            logger.debug(f"Successfully added indicator columns: {added_cols}")
            if len(added_cols) < len(expected_cols) - 3:  # Allow MACD/BBands variations
                logger.warning(
                    "Not all expected indicators were added. Check data length and indicator parameters."
                )

            return df_out.round(4)  # Round for cleaner display

        except AttributeError as e:
            # Handle cases where df might not have 'ta' accessor if pandas_ta isn't properly installed/imported
            if "'DataFrame' object has no attribute 'ta'" in str(e):
                logger.error(
                    "Pandas TA extension not available on DataFrame. Is pandas_ta installed correctly?",
                    exc_info=True,
                )
                return df  # Return original df
            else:
                logger.error(
                    f"Attribute error calculating indicators: {e}", exc_info=True
                )
                return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return df  # Return original df on error


# --- Terminal UI ---


class TerminalUI:
    """Handles the terminal user interface, menus, and input."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager
        self.colors = config_manager.theme_colors

    def clear_screen(self) -> None:
        """Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def display_header(self, title: str) -> None:
        """Displays a standardized header."""
        self.clear_screen()
        try:
            term_width = os.get_terminal_size().columns
        except OSError:  # Handle cases where terminal size cannot be determined (e.g., running non-interactively)
            term_width = 80
        "=" * term_width

    def display_menu(
        self, title: str, options: list[str], prompt: str = "Enter your choice"
    ) -> str:
        """Displays a menu and gets user choice."""
        self.display_header(title)
        for _i, _option in enumerate(options, 1):
            pass

        while True:
            choice = input(
                f"\n{self.colors['input_prompt']}{prompt} (1-{len(options)}): {self.colors['reset']}"
            )
            choice = choice.strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                logger.debug(f"Menu '{title}' choice: {choice}")
                return choice
            else:
                self.print_error(
                    f"Invalid input. Please enter a number between 1 and {len(options)}."
                )
                time.sleep(1)  # Give user time to see error before input repeats

    def get_input(
        self,
        prompt: str,
        default: str | None = None,
        required: bool = True,
        input_type: type = str,
        validation_func: callable | None = None,
    ) -> Any:
        """Gets validated user input."""
        while True:
            prompt_full = f"{self.colors['input_prompt']}{prompt}"
            if default is not None:
                prompt_full += f" [default: {default}]"
            prompt_full += f": {self.colors['reset']}"
            user_input = input(prompt_full).strip()

            if not user_input and default is not None:
                user_input = default
                logger.debug(f"Input for '{prompt}' using default: {default}")

            if required and not user_input:
                self.print_error("Input is required.")
                continue

            if not user_input and not required:
                return None  # Allow empty optional input

            try:
                # Handle boolean conversion specifically if needed
                if input_type == bool:
                    if user_input.lower() in ("true", "1", "t", "y", "yes"):
                        value = True
                    elif user_input.lower() in ("false", "0", "f", "n", "no"):
                        value = False
                    else:
                        raise ValueError(
                            "Invalid boolean value. Use True/False, Yes/No, 1/0."
                        )
                else:
                    value = input_type(user_input)  # Convert to desired type

                if validation_func:
                    validation_result = validation_func(value)
                    # Allow validation func to return error message string
                    if isinstance(validation_result, str):
                        self.print_error(validation_result)
                        continue
                    # Allow validation func to return False for generic failure
                    elif validation_result is False:
                        self.print_error("Input validation failed.")
                        continue
                    # If validation_result is True, proceed

                logger.debug(f"Input for '{prompt}': {value} (type: {input_type})")
                return value
            except ValueError as e:
                self.print_error(
                    f"Invalid input type or value. Expected {input_type.__name__}. Details: {e}"
                )
            except Exception as e:
                self.print_error(f"Input validation error: {e}")

    def print_error(self, message: str) -> None:
        """Prints an error message."""
        logger.error(message)  # Also log it

    def print_success(self, message: str) -> None:
        """Prints a success message."""

    def print_warning(self, message: str) -> None:
        """Prints a warning message."""

    def print_info(self, message: str) -> None:
        """Prints an informational message."""

    def print_table(self, data: pd.DataFrame | dict, title: str | None = None) -> None:
        """Prints data in a formatted way (Pandas DataFrame or Dict)."""
        if title:
            pass

        if isinstance(data, pd.DataFrame):
            if data.empty:
                pass
            else:
                # Configure pandas display options for terminal
                try:
                    term_width = os.get_terminal_size().columns
                except OSError:
                    term_width = 100  # Default width if terminal size unavailable
                pd.set_option("display.max_rows", 50)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", term_width)
                pd.set_option(
                    "display.expand_frame_repr", True
                )  # Prevent wrapping lines
                pd.set_option(
                    "display.float_format", "{:.4f}".format
                )  # Adjust precision as needed
                # Use to_markdown for better alignment in many terminals
                try:
                    pass
                except (
                    ImportError
                ):  # Fallback if tabulate (markdown dep) isn't installed
                    pass
        elif isinstance(data, dict):
            if not data:
                pass
            else:
                max(len(str(k)) for k in data) if data else 0
                for _key, value in data.items():
                    # Color formatting based on value type or sign could be added here
                    self.colors["neutral"]
                    if isinstance(value, (float, np.floating, int)):
                        # Format floats nicely, use default for int
                        f"{value:.4f}" if isinstance(
                            value, (float, np.floating)
                        ) else str(value)
                        if value > 0:
                            self.colors["positive"]
                        elif value < 0:
                            self.colors["negative"]
                    else:
                        str(value)

        else:
            self.print_error("Unsupported data type for print_table.")

    def wait_for_enter(self, prompt: str = "Press Enter to continue...") -> None:
        """Pauses execution until Enter is pressed."""
        input(f"\n{self.colors['accent']}{prompt}{self.colors['reset']}")

    def display_chart(self, data: list[float], title: str) -> None:
        """Displays a simple ASCII chart."""
        if not data:
            self.print_warning(f"No data available for chart: {title}")
            return

        chart_height = self.config_manager.config.get("chart_height", 10)
        try:
            # Ensure data is list of numbers, handle potential NaN/inf
            # Replace non-finite with previous valid value or 0 if at start
            plot_data = []
            last_valid = 0
            for d in data:
                if np.isfinite(d):
                    plot_data.append(float(d))
                    last_valid = float(d)
                else:
                    plot_data.append(
                        last_valid
                    )  # Use last valid point to avoid zeroing dips

            if not plot_data:
                self.print_warning(f"No valid numeric data points for chart: {title}")
                return
            asciichart.plot(plot_data, {"height": chart_height})
        except Exception as e:
            self.print_error(f"Failed to generate chart: {e}")
            logger.error(f"Asciichart error for title '{title}': {e}", exc_info=True)


# --- Main Trading Terminal Application ---


class TradingTerminal:
    """Main class for the trading terminal application."""

    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.ui = TerminalUI(self.config_manager)
        self.credentials: APICredentials | None = None
        self.exchange_client: BybitFuturesCCXTClient | None = None
        self.ta_analyzer = TechnicalAnalysis(self.config_manager.config)
        self._running = False
        self._shutdown_event = asyncio.Event()  # Event to signal shutdown
        self._tasks: list[asyncio.Task] = []  # To keep track of background tasks if any

    def _setup_signal_handlers(self) -> None:
        """Sets up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                # Use lambda to ensure shutdown is called as a coroutine
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self.shutdown(signal=s))
                )
                logger.debug(f"Signal handler set for {sig.name}")
            except NotImplementedError:
                # Windows asyncio loop might not support add_signal_handler fully
                logger.warning(
                    f"Signal handler for {sig.name} not fully supported on this platform. Use Ctrl+C or Exit menu."
                )
                # Fallback or alternative mechanism might be needed for Windows clean exit on Ctrl+C
                # For now, rely on KeyboardInterrupt in main or Exit menu choice
                pass
        logger.info("Signal handlers set up (or attempted).")

    async def setup_credentials(self) -> bool:
        """Sets up API credentials from environment variables."""
        load_dotenv()
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        testnet_str = os.getenv("TESTNET", "False")
        testnet = testnet_str.lower() in ("true", "1", "t", "y", "yes")

        if (
            not api_key
            or not api_secret
            or api_key == "your_api_key_here"
            or api_secret == "your_api_secret_here"
        ):
            logger.error(
                "API credentials (BYBIT_API_KEY, BYBIT_API_SECRET) not found or not set in .env file."
            )
            self.ui.print_error(
                "API credentials not found or not configured in .env file."
            )
            self.ui.print_warning(
                "Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are correctly set in the .env file."
            )
            self.credentials = None
            return False
        else:
            logger.info(f"API credentials loaded successfully. Testnet mode: {testnet}")
            self.credentials = APICredentials(api_key, api_secret, testnet)
            return True

    async def initialize(self) -> None:
        """Initializes the terminal, including API client."""
        if not await self.setup_credentials():
            self.ui.print_warning(
                "Running in limited mode without valid API credentials."
            )
            # Allow running, but authenticated features will fail later if attempted.
            return

        # Credentials seem valid, initialize the client
        self.exchange_client = BybitFuturesCCXTClient(
            self.credentials, self.config_manager.config
        )
        try:
            # We need the client instance later, so we call initialize directly here
            # and rely on the shutdown method to close it.
            await self.exchange_client.initialize()
        except ConnectionError as e:
            self.ui.print_error(f"Failed to initialize exchange client: {e}")
            self.ui.print_warning(
                "Proceeding without a fully functional exchange client. Authenticated features will fail."
            )
            # Invalidate the client if initialization failed
            self.exchange_client = None
        except Exception as e:
            self.ui.print_error(f"Unexpected error during initialization: {e}")
            logger.critical("Critical initialization error", exc_info=True)
            # Trigger shutdown immediately on critical init failure
            await self.shutdown(exit_code=1)

    async def shutdown(self, signal=None, exit_code=0) -> None:
        """Gracefully shuts down the application."""
        if not self._running:
            # Avoid multiple shutdown calls
            if signal:
                logger.warning(
                    f"Shutdown already in progress or completed, ignoring signal {getattr(signal, 'name', signal)}"
                )
            return
        self._running = False  # Prevent new actions in the main loop

        signal_name = (
            f"signal {getattr(signal, 'name', signal)}" if signal else "request"
        )
        logger.info(f"Shutdown initiated by {signal_name}...")
        self.ui.print_info("\nShutting down terminal...")  # Notify user

        # Cancel any running background tasks (if any were implemented)
        if self._tasks:
            logger.info(f"Cancelling {len(self._tasks)} background tasks...")
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            # Wait for tasks to finish cancelling
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    logger.debug(f"Task {i} cancelled successfully.")
                elif isinstance(result, Exception):
                    logger.error(
                        f"Error during background task cancellation/shutdown: {result}",
                        exc_info=result,
                    )
            logger.info("Background tasks processed.")

        # Close the exchange client connection
        if self.exchange_client:
            logger.info("Closing exchange client...")
            await self.exchange_client.close()  # Ensure close is called

        logger.info("Terminal shutdown complete.")
        self.ui.clear_screen()  # Clear screen one last time
        self.ui.print_info("Terminal exited.")

        # Set the shutdown event to allow the main loop to exit cleanly
        self._shutdown_event.set()

        # Note: We don't call sys.exit here directly. The main function will exit
        # after asyncio.run() completes, which happens when the main task finishes
        # (after _shutdown_event is set).

    async def run(self) -> None:
        """Runs the main terminal loop."""
        self._setup_signal_handlers()
        await self.initialize()

        # If initialization failed critically and triggered shutdown, exit early
        if self._shutdown_event.is_set():
            logger.warning(
                "Shutdown triggered during initialization, exiting run loop."
            )
            return

        self._running = True

        while self._running:
            # Use asyncio.wait to allow interruption by the shutdown event
            # Create the menu display and input task
            menu_task = asyncio.create_task(self.display_and_get_menu_choice())

            # Wait for either the menu choice or the shutdown signal
            done, pending = await asyncio.wait(
                [menu_task, asyncio.create_task(self._shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks (if menu finished, cancel shutdown wait; if shutdown, cancel menu)
            for task in pending:
                task.cancel()
                try:
                    await task  # Allow cancellation to propagate
                except asyncio.CancelledError:
                    pass  # Expected

            # Check if shutdown was triggered
            if self._shutdown_event.is_set():
                logger.info("Shutdown event detected, exiting main loop.")
                break  # Exit the loop cleanly

            # If menu task completed, get the result
            choice = None
            for task in done:
                if task == menu_task:
                    try:
                        choice = await task  # Get result if task finished normally
                    except asyncio.CancelledError:
                        logger.warning("Menu task was cancelled.")
                        # This might happen if shutdown occurred exactly during input, loop will exit on next iteration
                        continue
                    except Exception as e:
                        logger.error(f"Error in menu task: {e}", exc_info=True)
                        self.ui.print_error(f"An error occurred in the menu: {e}")
                        await asyncio.sleep(2)  # Pause for user
                        continue  # Go to next loop iteration
                    break  # Found the menu task result

            if choice is None:
                # Should not happen unless shutdown occurred, handled above
                continue

            # Handle the choice
            exit_choice = "5"  # Corresponds to "Exit" in the menu options
            if choice == exit_choice:
                # Initiate shutdown but let the loop terminate naturally via the event
                if (
                    self._running
                ):  # Avoid calling shutdown again if already called by signal
                    asyncio.create_task(self.shutdown())  # Start shutdown process
                # Loop will break on next iteration due to _shutdown_event being set
            else:
                # Handle other menu choices
                await self.handle_menu_choice(choice)
                # Small delay perhaps needed if actions are very fast, but input usually handles pacing
                # await asyncio.sleep(0.1)

        # Ensure final cleanup message after loop exits
        logger.info("Main run loop finished.")

    async def display_and_get_menu_choice(self) -> str:
        """Displays the menu and handles input asynchronously."""
        # This needs to be async to be awaitable in asyncio.wait
        # However, input() is blocking. We run it in an executor.
        loop = asyncio.get_running_loop()
        choice = await loop.run_in_executor(
            None,  # Use default thread pool executor
            self.ui.display_menu,  # The blocking function
            f"Bybit Futures Terminal {'(Testnet)' if self.credentials and self.credentials.testnet else ''}",  # arg 1
            [  # arg 2 (options)
                "Place Order",
                "View Account Balance",
                "Technical Analysis",
                "Settings",
                "Exit",
            ],
            "Enter your choice",  # arg 3 (prompt)
        )
        return choice

    async def handle_menu_choice(self, choice: str) -> None:
        """Handles user menu choices."""
        action = None
        requires_auth = False  # Flag actions needing credentials/client

        if choice == "1":
            action = self.place_order_menu
            requires_auth = True
        elif choice == "2":
            action = self.view_balance
            requires_auth = True
        elif choice == "3":
            action = self.technical_analysis_menu
            # TA might work with public data, but needs client init for fetching
            requires_auth = True  # Requires initialized client at least
        elif choice == "4":
            action = self.settings_menu
            requires_auth = False
        # Choice "5" (Exit) is handled in the main loop now
        else:
            # This case should technically not be reached due to menu validation
            self.ui.print_error("Invalid choice detected.")
            await asyncio.sleep(1)  # Short pause for user to see error
            return  # No action to perform

        # Check if authentication/client is required and available
        if requires_auth and (
            not self.credentials
            or not self.exchange_client
            or not self.exchange_client._initialized
        ):
            self.ui.print_error(
                "This action requires initialized API credentials and a connection."
            )
            self.ui.print_warning(
                "Please check your .env file and ensure the connection was successful on startup."
            )
            await self.ui_wait_for_enter_async()  # Use async version
            return

        if action:
            try:
                await action()
            except ConnectionError as e:
                self.ui.print_error(
                    f"Connection Error: {e}. Client might be offline or not initialized."
                )
                await self.ui_wait_for_enter_async()
            except ccxt.AuthenticationError as e:
                self.ui.print_error(
                    f"Authentication Error: {e}. Check API keys and permissions in Bybit."
                )
                await self.ui_wait_for_enter_async()
            except ccxt.NetworkError as e:
                self.ui.print_error(
                    f"Network Error: {e}. Check internet connection or Bybit status."
                )
                await self.ui_wait_for_enter_async()
            except ccxt.InsufficientFunds as e:
                self.ui.print_error(f"Insufficient Funds: {e}")
                await self.ui_wait_for_enter_async()
            except ccxt.InvalidOrder as e:
                self.ui.print_error(
                    f"Invalid Order: {e}. Check parameters (size, price, symbol)."
                )
                await self.ui_wait_for_enter_async()
            except ccxt.ExchangeError as e:  # Catch other specific exchange errors
                self.ui.print_error(f"Exchange Error: {e}.")
                await self.ui_wait_for_enter_async()
            except ValueError as e:  # Catch validation errors etc.
                self.ui.print_error(f"Input Error: {e}")
                await self.ui_wait_for_enter_async()
            except Exception as e:
                logger.error(
                    f"Unhandled error in menu action {action.__name__}: {e}",
                    exc_info=True,
                )
                self.ui.print_error(f"An unexpected error occurred: {e}")
                await self.ui_wait_for_enter_async()

    # --- Async Input Helpers ---
    # Wrap blocking UI input calls in run_in_executor to avoid blocking the event loop

    async def ui_get_input_async(self, *args, **kwargs) -> Any:
        """Asynchronously gets validated user input using executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.ui.get_input, *args, **kwargs)

    async def ui_wait_for_enter_async(
        self, prompt: str = "Press Enter to continue..."
    ) -> None:
        """Asynchronously waits for Enter key using executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.ui.wait_for_enter, prompt)

    # --- Input Validation Helpers ---

    def _validate_symbol(self, symbol: str) -> bool | str:
        """Validates symbol format. Returns True or error message string."""
        if not symbol:
            return "Symbol cannot be empty."
        symbol = symbol.upper()
        # Basic check: Should contain letters and possibly a slash or numbers
        if not any(c.isalpha() for c in symbol):
            return "Symbol must contain letters."

        # Check for common invalid characters (adjust as needed)
        invalid_chars = " !@#$%^&*()[]{};:'\"\\|,.<>?~`"
        if any(c in invalid_chars for c in symbol):
            return f"Symbol contains invalid characters. Found: {[c for c in symbol if c in invalid_chars]}"

        # Check format if slash is present
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                return "Invalid format. Use BASE/QUOTE (e.g., BTC/USDT)."
        # Check length for non-slash format (e.g., BTCUSDT)
        elif (
            len(symbol) < 4 or len(symbol) > 20
        ):  # Allow longer symbols like 1000PEPEUSDT etc.
            return "Symbol length seems unusual. Use format like BTCUSDT or BTC/USDT."

        # Optional: More robust validation against loaded markets (if client initialized)
        # Requires markets to be loaded successfully during init
        if (
            self.exchange_client
            and self.exchange_client.exchange
            and self.exchange_client.exchange.markets
        ):
            ccxt_symbol = self._get_ccxt_symbol(symbol)  # Ensure it's in CCXT format
            if ccxt_symbol not in self.exchange_client.exchange.markets:
                # List some available symbols if possible
                available_symbols = list(self.exchange_client.exchange.markets.keys())[
                    :5
                ]
                return f"Symbol '{ccxt_symbol}' not found on loaded exchange markets. Available examples: {', '.join(available_symbols)}..."

        return True

    def _get_ccxt_symbol(self, user_input: str) -> str:
        """Converts user input (BTCUSDT or BTC/USDT) to CCXT format (BTC/USDT)."""
        symbol = user_input.strip().upper()
        if "/" not in symbol:
            # Attempt common conversions (USDT, BUSD, USDC, BTC, ETH) - prioritize USDT for Bybit Futures
            # Use a more robust check based on loaded markets if available
            if (
                self.exchange_client
                and self.exchange_client.exchange
                and self.exchange_client.exchange.markets
            ):
                possible_matches = []
                for market_symbol in self.exchange_client.exchange.markets:
                    base, quote = market_symbol.split("/")
                    if symbol == f"{base}{quote}":
                        possible_matches.append(market_symbol)
                if len(possible_matches) == 1:
                    formatted = possible_matches[0]
                    logger.debug(
                        f"Formatted symbol '{symbol}' to '{formatted}' based on markets."
                    )
                    return formatted
                elif len(possible_matches) > 1:
                    logger.warning(
                        f"Ambiguous symbol '{symbol}'. Matches: {possible_matches}. Returning first match: {possible_matches[0]}"
                    )
                    return possible_matches[0]
                # else: fall through to generic check

            # Generic check if markets not loaded or no match found
            for quote in ["USDT", "USD", "BUSD", "USDC", "BTC", "ETH", "DAI"]:
                if symbol.endswith(quote) and len(symbol) > len(quote):
                    base = symbol[: -len(quote)]
                    formatted = f"{base}/{quote}"
                    logger.debug(
                        f"Formatted symbol '{symbol}' to '{formatted}' (generic)."
                    )
                    return formatted
            # If no common quote found, log warning and return as is
            logger.warning(
                f"Could not automatically format symbol '{symbol}' to BASE/QUOTE. Using as is. CCXT might reject it."
            )
        return symbol  # Return original or formatted symbol (if already has /)

    def _validate_side(self, side: str) -> bool | str:
        if side.lower() not in ["buy", "sell"]:
            return "Invalid side. Must be 'buy' or 'sell'."
        return True

    def _validate_order_type(self, order_type: str) -> bool | str:
        # CCXT might support more, but limit to common ones for simplicity in UI
        supported_types = [
            "market",
            "limit",
        ]  # Add more if needed: 'Stop', 'TakeProfit', 'StopLimit' etc.
        if order_type.lower() not in supported_types:
            return f"Invalid order type. Use {' or '.join(t.capitalize() for t in supported_types)}."
        return True

    def _validate_positive_float(self, value: Any) -> bool | str:
        # Input might not be float yet if validation runs before type conversion
        try:
            num_value = float(value)
            if num_value <= 0:
                return "Value must be a positive number."
            return True
        except (ValueError, TypeError):
            return "Input must be a valid number."

    def _validate_timeframe(self, timeframe: str) -> bool | str:
        """Basic validation for common CCXT timeframe formats."""
        # Example: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        # This is not exhaustive but covers many cases. CCXT handles specifics.
        import re

        if not timeframe:
            return "Timeframe cannot be empty."
        if not re.match(r"^\d+[mhdwyM]$", timeframe):
            return "Invalid timeframe format. Use like '1m', '1h', '1d', '1w'."
        # Optional: Check against exchange.timeframes if markets are loaded
        if (
            self.exchange_client
            and self.exchange_client.exchange
            and self.exchange_client.exchange.timeframes
        ):
            if timeframe not in self.exchange_client.exchange.timeframes:
                # List some available timeframes if possible
                available_tfs = list(self.exchange_client.exchange.timeframes.keys())[
                    :10
                ]
                return f"Timeframe '{timeframe}' may not be supported by the exchange. Available examples: {', '.join(available_tfs)}..."
        return True

    # --- Menu Actions ---

    async def place_order_menu(self) -> None:
        """Handles the logic for placing an order via user input."""
        self.ui.display_header("Place Order")

        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        # Use async input getter
        symbol_input = await self.ui_get_input_async(
            "Enter symbol (e.g., BTC/USDT or BTCUSDT)",
            default=default_symbol,
            validation_func=self._validate_symbol,
        )
        symbol = self._get_ccxt_symbol(symbol_input)  # Convert to CCXT format

        side = await self.ui_get_input_async(
            "Enter side (buy/sell)", validation_func=self._validate_side
        )
        side = side.lower()

        default_order_type = self.config_manager.config.get(
            "default_order_type", "Limit"
        )
        order_type = await self.ui_get_input_async(
            "Enter order type (Market/Limit)",
            default=default_order_type,
            validation_func=self._validate_order_type,
        )
        order_type = order_type.lower()  # Use lower case for CCXT consistency

        amount = await self.ui_get_input_async(
            "Enter quantity",
            input_type=float,
            validation_func=self._validate_positive_float,
        )

        price = None
        params = {}  # Dictionary for extra parameters like stopLoss, takeProfit, reduceOnly etc.

        if order_type == "limit":
            price = await self.ui_get_input_async(
                "Enter price",
                input_type=float,
                validation_func=self._validate_positive_float,
            )

        # --- Optional Advanced Order Parameters ---
        add_advanced = await self.ui_get_input_async(
            "Add advanced options (StopLoss/TakeProfit/ReduceOnly)? (yes/no)",
            default="no",
            required=False,
        )
        if add_advanced and add_advanced.lower() == "yes":
            # Stop Loss
            sl_price = await self.ui_get_input_async(
                "Enter Stop Loss price (0 or leave blank for none)",
                default="0",
                required=False,
                input_type=float,
                validation_func=lambda x: isinstance(x, (float, int))
                and x >= 0
                or "Must be a non-negative number",
            )
            if sl_price is not None and sl_price > 0:
                params["stopLoss"] = {
                    "type": "market",
                    "triggerPrice": sl_price,
                }  # Bybit often uses triggerPrice for SL/TP
                # Could add options for limit SL: params['stopLoss'] = {'type': 'limit', 'triggerPrice': sl_price, 'price': sl_limit_price}
                self.ui.print_info(f"Stop Loss trigger set at {sl_price}")

            # Take Profit
            tp_price = await self.ui_get_input_async(
                "Enter Take Profit price (0 or leave blank for none)",
                default="0",
                required=False,
                input_type=float,
                validation_func=lambda x: isinstance(x, (float, int))
                and x >= 0
                or "Must be a non-negative number",
            )
            if tp_price is not None and tp_price > 0:
                params["takeProfit"] = {"type": "market", "triggerPrice": tp_price}
                self.ui.print_info(f"Take Profit trigger set at {tp_price}")

            # Reduce Only
            reduce_only = await self.ui_get_input_async(
                "Set as Reduce Only? (yes/no)", default="no", required=False
            )
            if reduce_only and reduce_only.lower() == "yes":
                params["reduceOnly"] = True
                self.ui.print_info("Order set to Reduce Only.")

            # Time In Force (Example)
            # tif = await self.ui_get_input_async("Enter Time In Force (GTC/IOC/FOK, default GTC)", default="GTC", required=False)
            # if tif and tif.upper() in ['GTC', 'IOC', 'FOK']:
            #     params['timeInForce'] = tif.upper()

        # Confirmation
        confirm_msg = f"Confirm: Place {order_type.upper()} {side.upper()} order for {amount} {symbol}"
        if price:
            confirm_msg += f" at price {price}"
        if params.get("stopLoss"):
            confirm_msg += f" with SL @ {params['stopLoss']['triggerPrice']}"
        if params.get("takeProfit"):
            confirm_msg += f" with TP @ {params['takeProfit']['triggerPrice']}"
        if params.get("reduceOnly"):
            confirm_msg += " (Reduce Only)"

        confirm_msg += "?"
        confirm = await self.ui_get_input_async("Type 'yes' to confirm", default="no")

        if confirm.lower() != "yes":
            self.ui.print_warning("Order cancelled.")
            await self.ui_wait_for_enter_async()
            return

        # Place the order via CCXT client
        # Exceptions are caught by handle_menu_choice
        self.ui.print_info("Submitting order...")
        result = await self.exchange_client.place_order(
            symbol, side, order_type, amount, price, params
        )

        order_id = result.get("id", "N/A")
        status = result.get("status", "N/A")
        filled = result.get("filled", 0.0)
        avg_price = result.get("average")

        success_msg = f"Order submitted! ID: {order_id}, Status: {status}"
        if filled > 0:
            success_msg += f", Filled: {filled}"
            if avg_price:
                success_msg += f" @ avg price {avg_price:.4f}"
        self.ui.print_success(success_msg)

        # Display more details from the result dict in a table
        # Filter result for display - include relevant fields
        display_keys = [
            "id",
            "datetime",
            "symbol",
            "type",
            "side",
            "price",
            "amount",
            "cost",
            "filled",
            "remaining",
            "average",
            "status",
            "fee",
            "stopLossPrice",
            "takeProfitPrice",
            "reduceOnly",
        ]
        display_result = {
            k: v for k, v in result.items() if k in display_keys and v is not None
        }  # Filter keys and remove None values
        self.ui.print_table(display_result, title="Order Submission Result")

        await self.ui_wait_for_enter_async()

    async def view_balance(self) -> None:
        """Displays account balances using CCXT."""
        self.ui.display_header("Account Balance (USDT Futures)")
        self.ui.print_info("Fetching balance...")

        # Exceptions are caught by handle_menu_choice
        balance_data = (
            await self.exchange_client.fetch_balance()
        )  # Already gets USDT part

        if not balance_data:
            self.ui.print_warning("No USDT balance data found in response.")
        else:
            # Prepare data for display - use standardized CCXT keys
            display_data = {
                "Available": balance_data.get("free", 0.0),
                "Used Margin": balance_data.get("used", 0.0),
                "Total Equity": balance_data.get("total", 0.0),
                # Currency is implicitly USDT here
            }
            # Try to get PNL from the raw 'info' if available (exchange-specific)
            if "info" in balance_data and isinstance(balance_data["info"], dict):
                raw_info = balance_data["info"]
                # Bybit v5 API balance endpoint structure (example field names)
                unrealized_pnl = raw_info.get(
                    "unrealisedPnl"
                )  # Check this exact key in Bybit docs/response
                realized_pnl = raw_info.get("cumRealisedPnl")  # Check this exact key
                equity = raw_info.get("equity")  # Cross-check total equity

                if unrealized_pnl is not None:
                    try:
                        display_data["Unrealized PNL (All Pos)"] = float(unrealized_pnl)
                    except ValueError:
                        pass  # Ignore if not a number
                if realized_pnl is not None:
                    with contextlib.suppress(ValueError):
                        display_data["Realized PNL (Session)"] = float(realized_pnl)
                # Sometimes total is calculated differently, show raw equity if available
                if equity is not None and "Total Equity" in display_data:
                    with contextlib.suppress(ValueError):
                        display_data["Raw Equity"] = float(equity)

            self.ui.print_table(display_data, title="USDT Balance")

            # Suggest fetching positions for more detail
            self.ui.print_info(
                "\nNote: PNL figures might be estimates based on balance data."
            )
            self.ui.print_info(
                "For precise PNL per position, fetch open positions (feature not yet implemented)."
            )

        await self.ui_wait_for_enter_async()

    async def technical_analysis_menu(self) -> None:
        """Handles fetching data and displaying technical analysis."""
        self.ui.display_header("Technical Analysis")

        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        symbol_input = await self.ui_get_input_async(
            "Enter symbol (e.g., BTC/USDT or BTCUSDT)",
            default=default_symbol,
            validation_func=self._validate_symbol,
        )
        symbol = self._get_ccxt_symbol(symbol_input)

        default_timeframe = self.config_manager.config.get("default_timeframe", "1h")
        timeframe = await self.ui_get_input_async(
            "Enter timeframe (e.g., 1m, 5m, 1h, 1d)",
            default=default_timeframe,
            validation_func=self._validate_timeframe,
        )

        # Fetch OHLCV data
        self.ui.print_info(f"Fetching OHLCV data for {symbol} ({timeframe})...")

        # Fetch enough data for longest indicator + chart points
        chart_points = self.config_manager.config.get("chart_points", 50)
        # Estimate max lookback needed for indicators (e.g., MACD needs ~34, BBands default 20, add buffer)
        indicator_periods = self.config_manager.config.get("indicator_periods", {})
        indicator_lookback = max(
            indicator_periods.get("sma", 20),
            indicator_periods.get("ema", 20),
            indicator_periods.get("rsi", 14)
            + 1,  # RSI needs period + 1 for initial diff
            indicator_periods.get("bbands", 20),
            26 + 9,  # MACD typical lookback (26 for slow EMA, 9 for signal)
        )
        # Add buffer for calculations and potential missing data points
        fetch_limit = max(chart_points, indicator_lookback) + 100  # Generous buffer
        logger.debug(f"Fetching {fetch_limit} candles for TA/Chart.")

        # Exceptions are caught by handle_menu_choice
        ohlcv_data = await self.exchange_client.fetch_ohlcv(
            symbol, timeframe, limit=fetch_limit
        )

        if not ohlcv_data:
            self.ui.print_warning("No OHLCV data received.")
            await self.ui_wait_for_enter_async()
            return

        # Convert to DataFrame
        try:
            df = pd.DataFrame(
                ohlcv_data,
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df.set_index("Timestamp", inplace=True)
            # Ensure numeric types after potential issues
            df = df.apply(pd.to_numeric, errors="coerce")
            df.dropna(
                subset=["Close"], inplace=True
            )  # Drop rows where close is missing

        except Exception as e:
            logger.error(
                f"Error processing OHLCV data into DataFrame: {e}", exc_info=True
            )
            self.ui.print_error(f"Failed to process market data: {e}")
            await self.ui_wait_for_enter_async()
            return

        if df.empty:
            self.ui.print_warning("DataFrame is empty after processing OHLCV data.")
            await self.ui_wait_for_enter_async()
            return

        # Calculate indicators
        df_indicators = self.ta_analyzer.calculate_indicators(df)

        # Display chart of close prices
        close_prices = df_indicators["Close"].tail(chart_points).tolist()
        self.ui.display_chart(
            close_prices,
            f"{symbol} ({timeframe}) Close Price Trend (Last {chart_points} points)",
        )

        # Display table of recent indicator values (last 10 rows)
        # Dynamically find indicator columns added by pandas_ta
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        indicator_cols = [
            col
            for col in df_indicators.columns
            if col not in base_cols and col != "Timestamp"
        ]  # Exclude original + timestamp

        # Prepare DataFrame for display
        display_df = df_indicators.tail(10).reset_index()  # Keep timestamp for display
        display_df["Timestamp"] = display_df["Timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Format timestamp

        # Define column order for display
        ordered_cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"] + sorted(
            indicator_cols
        )
        # Select only existing columns from our desired list
        existing_cols_to_show = [
            col for col in ordered_cols if col in display_df.columns
        ]

        self.ui.print_table(
            display_df[existing_cols_to_show], title="Recent Market Data & Indicators"
        )

        await self.ui_wait_for_enter_async()

    async def settings_menu(self) -> None:
        """Handles the settings menu."""
        # Use a local running flag for this menu loop
        settings_running = True
        while settings_running and self._running:  # Check global running flag too
            current_config = self.config_manager.config
            options = [
                f"Change Theme (Current: {current_config.get('theme', 'N/A')})",
                f"Set Default Symbol (Current: {current_config.get('default_symbol', 'N/A')})",
                f"Set Default Timeframe (Current: {current_config.get('default_timeframe', 'N/A')})",
                f"Set Default Order Type (Current: {current_config.get('default_order_type', 'N/A')})",
                f"Set Log Level (Current: {current_config.get('log_level', 'N/A')})",
                "Back to Main Menu",
            ]
            # Use a synchronous menu display here as it's within an async function already
            # but the input itself needs to be async
            loop = asyncio.get_running_loop()
            choice = await loop.run_in_executor(
                None, self.ui.display_menu, "Settings", options
            )

            if not self._running:  # Check global flag again after input
                settings_running = False
                break

            if choice == "1":  # Change Theme
                theme = await self.ui_get_input_async(
                    "Enter theme (dark/light)", default=current_config.get("theme")
                )
                if theme and theme.lower() in ["dark", "light"]:
                    self.config_manager.config["theme"] = theme.lower()
                    self.config_manager.theme_colors = (
                        self.config_manager._setup_theme_colors()
                    )
                    self.ui.colors = (
                        self.config_manager.theme_colors
                    )  # Update UI instance colors
                    self.config_manager.save_config()
                    self.ui.print_success(f"Theme changed to {theme.lower()}.")
                else:
                    self.ui.print_error("Invalid theme. Choose 'dark' or 'light'.")
                await asyncio.sleep(1)

            elif choice == "2":  # Set Default Symbol
                symbol_input = await self.ui_get_input_async(
                    "Enter default symbol",
                    default=current_config.get("default_symbol"),
                    validation_func=self._validate_symbol,
                )
                if symbol_input:  # Ensure input was given (or default used)
                    symbol = self._get_ccxt_symbol(symbol_input)
                    self.config_manager.config["default_symbol"] = symbol
                    self.config_manager.save_config()
                    self.ui.print_success(f"Default symbol set to {symbol}.")
                await asyncio.sleep(1)

            elif choice == "3":  # Set Default Timeframe
                timeframe = await self.ui_get_input_async(
                    "Enter default timeframe",
                    default=current_config.get("default_timeframe"),
                    validation_func=self._validate_timeframe,
                )
                if timeframe:
                    self.config_manager.config["default_timeframe"] = timeframe
                    self.config_manager.save_config()
                    self.ui.print_success(f"Default timeframe set to {timeframe}.")
                await asyncio.sleep(1)

            elif choice == "4":  # Set Default Order Type
                order_type = await self.ui_get_input_async(
                    "Enter default order type (Market/Limit)",
                    default=current_config.get("default_order_type"),
                    validation_func=self._validate_order_type,
                )
                if order_type:
                    self.config_manager.config["default_order_type"] = (
                        order_type.capitalize()
                    )  # Store capitalized
                    self.config_manager.save_config()
                    self.ui.print_success(
                        f"Default order type set to {order_type.capitalize()}."
                    )
                await asyncio.sleep(1)

            elif choice == "5":  # Set Log Level
                log_level = await self.ui_get_input_async(
                    "Enter log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
                    default=current_config.get("log_level"),
                )
                if log_level and log_level.upper() in [
                    "DEBUG",
                    "INFO",
                    "WARNING",
                    "ERROR",
                    "CRITICAL",
                ]:
                    log_level_upper = log_level.upper()
                    self.config_manager.config["log_level"] = log_level_upper
                    self.config_manager._apply_log_level()  # Apply immediately
                    self.config_manager.save_config()
                    self.ui.print_success(f"Log level set to {log_level_upper}.")
                    logger.info(
                        f"Log level changed to {log_level_upper} via settings."
                    )  # Log the change
                else:
                    self.ui.print_error("Invalid log level.")
                await asyncio.sleep(1)

            elif choice == "6":  # Back
                settings_running = False  # Exit settings loop
            else:
                # Should not happen with validated menu
                self.ui.print_error("Invalid choice")
                await asyncio.sleep(1)


# --- Main Execution ---


async def main() -> None:
    """Main asynchronous function to run the trading terminal."""
    # Check/Create .env file
    env_path = Path(".env")
    if not env_path.exists():
        try:
            with open(env_path, "w") as f:
                f.write("# Bybit API Credentials (replace with your actual keys)\n")
                f.write("BYBIT_API_KEY=your_api_key_here\n")
                f.write("BYBIT_API_SECRET=your_api_secret_here\n\n")
                f.write(
                    "# Set to True to use Bybit's testnet environment (e.g., for testing)\n"
                )
                f.write("# Testnet URL: https://testnet.bybit.com\n")
                f.write("TESTNET=False\n")
            return  # Exit after creating the file, requiring user action
        except OSError:
            return  # Exit if file cannot be created

    # Run the terminal
    terminal = TradingTerminal()
    main_task = None
    try:
        # Create the main task so we can potentially await it after catching KeyboardInterrupt
        main_task = asyncio.create_task(terminal.run())
        await main_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled, likely during shutdown.")
        # Ensure cleanup runs if cancellation happened abruptly
        if terminal._running:  # Check if shutdown wasn't completed
            logger.warning(
                "Main task cancelled but terminal still marked as running. Forcing shutdown."
            )
            # Use create_task to avoid awaiting shutdown within the cancel handler
            asyncio.create_task(terminal.shutdown())
            # Give shutdown a moment to proceed
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        # This catch is mainly for platforms where signal handlers might not work perfectly
        # or if Ctrl+C is hit before the signal handler is fully registered.
        logger.warning("KeyboardInterrupt caught in main.")
        if terminal._running:
            # Manually trigger shutdown if the signal handler didn't catch it or isn't working
            # Use create_task as we are in an exception handler
            shutdown_task = asyncio.create_task(
                terminal.shutdown(signal="KeyboardInterrupt")
            )
            try:
                await asyncio.wait_for(
                    shutdown_task, timeout=5.0
                )  # Wait briefly for shutdown
            except TimeoutError:
                logger.error("Shutdown timed out after KeyboardInterrupt.")
            except Exception as e:
                logger.error(f"Error during shutdown after KeyboardInterrupt: {e}")
        # Wait for the main task to finish shutting down if it exists and isn't done
        if main_task and not main_task.done():
            logger.info("Waiting for main task to complete cancellation/shutdown...")
            try:
                await asyncio.wait_for(main_task, timeout=5.0)  # Wait briefly
            except TimeoutError:
                logger.error("Main task did not complete shutdown within timeout.")
            except asyncio.CancelledError:
                pass  # Expected if shutdown was successful
            except Exception as e:
                logger.error(f"Error waiting for main task completion: {e}")

    except Exception as e:
        logger.critical(
            f"Critical unhandled error in main execution: {e}", exc_info=True
        )
        # Attempt graceful shutdown even on critical error
        if terminal._running:
            logger.info("Attempting shutdown after critical error...")
            shutdown_task = asyncio.create_task(terminal.shutdown(exit_code=1))
            try:
                await asyncio.wait_for(shutdown_task, timeout=5.0)
            except Exception as shutdown_e:
                logger.error(
                    f"Error during shutdown after critical error: {shutdown_e}"
                )
    finally:
        # Ensure logs are flushed before exit
        logging.shutdown()


if __name__ == "__main__":
    # Check Python version (optional but good practice for async features)

    # No uvloop section anymore. Using standard asyncio.
    logging.info("Using default asyncio event loop.")

    # Run the main async function
    exit_code = 0
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch KeyboardInterrupt here if it happens *during* asyncio.run() setup/teardown
        # or if signal handlers failed entirely.
        logger.warning("KeyboardInterrupt caught outside main coroutine.")
        exit_code = 1
    except Exception as e:
        # Catch errors during asyncio.run() setup/teardown itself if any
        # Use a basic logger here as the main one might be shut down
        logging.getLogger("MainExec").critical(
            f"Fatal error outside main coroutine: {e}", exc_info=True
        )
        exit_code = 1
    finally:
        sys.exit(exit_code)
