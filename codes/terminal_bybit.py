#!/usr/bin/env python3
"""Bybit Futures Terminal - v2.8 - Pyrmethus Enhanced Edition
Original Author: Mentallyspammed1 (Enhanced by AI & Pyrmethus)
Last Updated: 2025-04-25 (Pyrmethus Weaving Complete) - Enhanced based on review & fixing observed errors.

Description:
A command-line interface for interacting with Bybit Futures (USDT Perpetual).
This enhanced version focuses on:
- Unified API interaction via the CCXT library for consistency and robustness (V5 focused).
- Improved error handling and user feedback, including specific CCXT exceptions.
- Correct asynchronous operations using standard asyncio with non-blocking input handling via executor.
- Configuration management via JSON and .env files with safe merging and normalization.
- Enhanced technical analysis: MAs, RSI, MACD, BBands, STOCH, ATR, Pivots, Trend Summary.
- Trailing Stop orders (percentage-based), Price Check, Open Positions/Orders view.
- Clear, color-coded terminal UI using Colorama and improved table formatting (tabulate).
- Robust ASCII charting for price trends with color.
- Fixed async input handling using asyncio's run_in_executor and functools.partial.
- Graceful shutdown handling via signals (SIGINT, SIGTERM) and menu option.
- Automatic .env file creation on first run with instructions.
- Robust input validation for symbols, timeframes, order types, quantities, etc.
- Dynamic menu options based on API connection status.
- Fixes for observed errors (DataFrame boolean ambiguity, V5 balance fetch, input validation).
"""

import asyncio
import json
import logging
import os
import re  # For timeframe and symbol validation
import signal  # For graceful shutdown
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial  # partial is used for async input handling
from pathlib import Path
from typing import Any

# Async & CCXT
import ccxt.async_support as ccxt
import numpy as np

# Data & Analysis
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None  # Set ta to None if not installed

# Configuration & Environment
import asciichartpy as asciichart

# UI & Utilities
from colorama import Fore, Style, init
from dotenv import load_dotenv

try:
    # Used for better table formatting in print_table
    from tabulate import tabulate
except ImportError:
    tabulate = None  # Set to None if not installed


# --- Initial Setup ---

# Initialize colorama for cross-platform terminal colors FIRST
init(autoreset=True)

# Configure logging
log_file = "terminal.log"
# Ensure log file directory exists (useful if run from different locations)
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # Default level, will be overridden by config later
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),  # Specify encoding
        logging.StreamHandler(sys.stdout),  # Log to stdout as well
    ],
)
logger = logging.getLogger("BybitTerminal")

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
        "default_order_type": "Limit",
        "connection_timeout_ms": 30000,
        "order_history_limit": 50,  # Note: Not yet used, kept for future
        "indicator_periods": {
            "sma_short": 20,
            "sma_long": 50,
            "ema": 20,
            "rsi": 14,
            "bbands": 20,
            "bbands_std": 2.0,
            "stoch_k": 14,
            "stoch_d": 3,
            "stoch_smooth_k": 3,
            "atr": 14,
            # MACD defaults (12, 26, 9) are standard in pandas_ta
        },
        "chart_height": 10,
        "chart_points": 60,
        "pivot_period": "1d",  # Timeframe for pivot point calculation (use lowercase CCXT standard)
    }

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config: dict = self._load_config()
        self._apply_log_level()  # Apply log level early
        self.theme_colors: dict = self._setup_theme_colors()
        logger.info(f"{Fore.CYAN}# Configuration spell cast from '{self.config_path}'{Style.RESET_ALL}")

    def _load_config(self) -> dict:
        """Loads configuration from JSON file or creates default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    # Ensure nested defaults are present if missing in loaded config
                    merged_config = self._deep_merge_dicts(self.DEFAULT_CONFIG.copy(), loaded_config)
                    # Normalize specific values like timeframe to lowercase standard
                    merged_config["default_timeframe"] = merged_config.get(
                        "default_timeframe", self.DEFAULT_CONFIG["default_timeframe"]
                    ).lower()
                    merged_config["pivot_period"] = merged_config.get(
                        "pivot_period", self.DEFAULT_CONFIG["pivot_period"]
                    ).lower()
                    merged_config["default_symbol"] = merged_config.get(
                        "default_symbol", self.DEFAULT_CONFIG["default_symbol"]
                    ).upper()
                    merged_config["default_order_type"] = merged_config.get(
                        "default_order_type", self.DEFAULT_CONFIG["default_order_type"]
                    ).capitalize()
                    merged_config["log_level"] = merged_config.get(
                        "log_level", self.DEFAULT_CONFIG["log_level"]
                    ).upper()
                    return merged_config
            except json.JSONDecodeError:
                logger.error(
                    f"{Fore.RED}Error decoding JSON from '{self.config_path}'. Conjuring default config.{Style.RESET_ALL}"
                )
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to load config: {e}. Conjuring default config.{Style.RESET_ALL}")
                return self.DEFAULT_CONFIG.copy()
        else:
            logger.warning(
                f"{Fore.YELLOW}'{self.config_path}' not found. Conjuring default configuration.{Style.RESET_ALL}"
            )
            return self._create_default_config()

    def _deep_merge_dicts(self, base: dict, update: dict) -> dict:
        """Recursively merges update dict into base dict."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = self._deep_merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    def _create_default_config(self) -> dict:
        """Creates a default configuration file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure defaults are normalized before writing
            default_config = self.DEFAULT_CONFIG.copy()
            default_config["default_timeframe"] = default_config["default_timeframe"].lower()
            default_config["pivot_period"] = default_config["pivot_period"].lower()
            default_config["default_symbol"] = default_config["default_symbol"].upper()
            default_config["default_order_type"] = default_config["default_order_type"].capitalize()
            default_config["log_level"] = default_config["log_level"].upper()

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            logger.info(f"{Fore.GREEN}Default configuration scroll inscribed at '{self.config_path}'{Style.RESET_ALL}")
            return default_config
        except OSError as e:
            logger.error(
                f"{Fore.RED}Error inscribing default config scroll: {e}. Using in-memory default.{Style.RESET_ALL}"
            )
            return self.DEFAULT_CONFIG.copy()

    def save_config(self) -> None:
        """Saves the current configuration to the file."""
        try:
            # Ensure specific keys are normalized before saving
            self.config["default_timeframe"] = self.config.get(
                "default_timeframe", self.DEFAULT_CONFIG["default_timeframe"]
            ).lower()
            self.config["pivot_period"] = self.config.get("pivot_period", self.DEFAULT_CONFIG["pivot_period"]).lower()
            self.config["default_symbol"] = self.config.get(
                "default_symbol", self.DEFAULT_CONFIG["default_symbol"]
            ).upper()
            self.config["default_order_type"] = self.config.get(
                "default_order_type", self.DEFAULT_CONFIG["default_order_type"]
            ).capitalize()
            self.config["log_level"] = self.config.get("log_level", self.DEFAULT_CONFIG["log_level"]).upper()

            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f"{Fore.GREEN}Configuration scroll updated at '{self.config_path}'{Style.RESET_ALL}")
        except OSError as e:
            logger.error(f"{Fore.RED}Error saving configuration scroll: {e}{Style.RESET_ALL}")

    def _apply_log_level(self) -> None:
        """Applies the log level from the configuration."""
        log_level_str = self.config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        root_logger = logging.getLogger()  # Get root logger
        root_logger.setLevel(log_level)  # Set root logger level first
        logger.setLevel(log_level)  # Set our specific logger level
        # Ensure handlers also respect the new level
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
        logger.info(f"{Fore.BLUE}# Log level attuned to {log_level_str}{Style.RESET_ALL}")

    def _setup_theme_colors(self) -> dict:
        """Sets up theme colors based on the configuration."""
        theme = self.config.get("theme", "dark").lower()
        if theme == "dark":
            # Dark theme with mystical hues
            return {
                "primary": Fore.CYAN,
                "secondary": Fore.MAGENTA,
                "accent": Fore.YELLOW,
                "error": Fore.RED + Style.BRIGHT,
                "success": Fore.GREEN + Style.BRIGHT,
                "warning": Fore.YELLOW,
                "info": Fore.BLUE + Style.BRIGHT,
                "title": Fore.CYAN + Style.BRIGHT,
                "menu_option": Fore.WHITE,
                "menu_highlight": Fore.YELLOW + Style.BRIGHT,
                "input_prompt": Fore.MAGENTA + Style.BRIGHT,
                "table_header": Fore.CYAN + Style.BRIGHT,
                "positive": Fore.GREEN,
                "negative": Fore.RED,
                "neutral": Fore.WHITE,
                "dim": Style.DIM,
                "reset": Style.RESET_ALL,
            }
        else:  # Light theme (Adjust as needed for better light mode visibility)
            return {
                "primary": Fore.BLUE,
                "secondary": Fore.GREEN,
                "accent": Fore.MAGENTA,
                "error": Fore.RED + Style.BRIGHT,
                "success": Fore.GREEN + Style.BRIGHT,
                "warning": Fore.YELLOW,
                "info": Fore.CYAN + Style.BRIGHT,
                "title": Fore.BLUE + Style.BRIGHT,
                "menu_option": Fore.BLACK,
                "menu_highlight": Fore.BLUE + Style.BRIGHT,
                "input_prompt": Fore.MAGENTA + Style.BRIGHT,
                "table_header": Fore.BLUE + Style.BRIGHT,
                "positive": Fore.GREEN,
                "negative": Fore.RED,
                "neutral": Fore.BLACK,
                "dim": Style.DIM,
                "reset": Style.RESET_ALL,
            }


# --- CCXT Exchange Client ---


class BybitFuturesCCXTClient:
    """Client for interacting with Bybit Futures via CCXT.
    Handles initialization, context management, and core API calls (V5 focus).
    """

    def __init__(self, credentials: APICredentials, config: dict) -> None:
        self.credentials = credentials
        self.config = config
        self.exchange: ccxt.bybit | None = None
        self._initialized = False
        self._markets_loaded = False

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Sets up the CCXT Bybit exchange instance, loads markets, handles V5."""
        if self._initialized:
            logger.debug("CCXT client already initialized.")
            return

        logger.info(f"{Fore.CYAN}# Summoning connection to Bybit via CCXT...{Style.RESET_ALL}")
        exchange_config = {
            "apiKey": self.credentials.api_key,
            "secret": self.credentials.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",  # Explicitly target perpetual swaps
                "adjustForTimeDifference": True,
                # V5 API requires 'category' often, CCXT might handle via defaultType/methods
                # but we'll add it explicitly in methods where needed.
                # 'defaultMarginMode': 'isolated', # Or 'cross'. Set if needed globally.
                # 'createOrderRequiresPrice': False, # Let CCXT handle this default
            },
            "timeout": self.config.get("connection_timeout_ms", 30000),
        }

        self.exchange = ccxt.bybit(exchange_config)  # Initialize first

        if self.credentials.testnet:
            logger.warning(f"{Fore.YELLOW}# Engaging Bybit TESTNET dimension.{Style.RESET_ALL}")
            # Prefer set_sandbox_mode for robustness
            if hasattr(self.exchange, "set_sandbox_mode"):
                try:
                    self.exchange.set_sandbox_mode(True)
                    logger.info(f"{Fore.BLUE}Using set_sandbox_mode(True) for Testnet.{Style.RESET_ALL}")
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Error calling set_sandbox_mode: {e}. Testnet may not function correctly.{Style.RESET_ALL}"
                    )
                    # Fallback might not be necessary if set_sandbox_mode is the intended way
            else:
                logger.warning(
                    f"{Fore.YELLOW}set_sandbox_mode not available, manually setting testnet API URL (may be less reliable).{Style.RESET_ALL}"
                )
                self.exchange.urls["api"] = self.exchange.urls["test"]

        try:
            logger.info(f"{Fore.BLUE}# Loading market contracts...{Style.RESET_ALL}")
            await self.exchange.load_markets()
            if not self.exchange.markets:
                logger.error(
                    f"{Fore.RED}Failed to load markets from Bybit. Check connection and API permissions.{Style.RESET_ALL}"
                )
                raise ConnectionError("Market loading failed - no markets returned")
            else:
                # Filter for USDT perpetual swaps (linear) if possible
                usdt_swap_markets = {
                    k: v
                    for k, v in self.exchange.markets.items()
                    if v.get("swap") and v.get("linear") and v.get("quote") == "USDT"
                }
                if not usdt_swap_markets:
                    logger.warning(
                        f"{Fore.YELLOW}No USDT Perpetual Swap markets found after loading. Check API key permissions or market availability.{Style.RESET_ALL}"
                    )
                logger.info(
                    f"{Fore.GREEN}Loaded {len(self.exchange.markets)} total markets, found {len(usdt_swap_markets)} USDT Perpetual Swaps.{Style.RESET_ALL}"
                )
                self._markets_loaded = True

            # Optional: Verify connectivity with a simple call after loading markets
            # await self.exchange.fetch_time() # Can add latency
            # logger.info("Connectivity test successful.")

            self._initialized = True
            logger.info(
                f"{Fore.GREEN}CCXT Bybit Futures client materialized successfully for {'Testnet' if self.credentials.testnet else 'Mainnet'}.{Style.RESET_ALL}"
            )

        except ccxt.AuthenticationError as e:
            logger.error(
                f"{Fore.RED}CCXT Authentication Sigil Rejected: Invalid API keys or permissions. Check .env file and Bybit API settings. {e}{Style.RESET_ALL}"
            )
            await self.close()  # Ensure cleanup on failure
            raise ConnectionError("CCXT Authentication Failed") from e
        except (TimeoutError, ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            logger.error(
                f"{Fore.RED}CCXT Network Rift: Could not connect to Bybit. Check internet and exchange status. {e}{Style.RESET_ALL}"
            )
            await self.close()
            raise ConnectionError("CCXT Network/Connection Failed") from e
        except ccxt.ExchangeError as e:  # Catch other exchange-specific errors during init/market load
            logger.error(f"{Fore.RED}CCXT Exchange Anomaly during initialization: {e}{Style.RESET_ALL}")
            await self.close()
            raise ConnectionError("CCXT Exchange Initialization Failed") from e
        except Exception as e:  # Catch any other unexpected error
            logger.error(
                f"{Fore.RED}An unexpected vortex occurred initializing CCXT: {e}{Style.RESET_ALL}", exc_info=True
            )
            await self.close()
            raise ConnectionError("Unexpected CCXT Initialization Error") from e

    async def close(self) -> None:
        """Closes the underlying CCXT exchange connection gracefully."""
        if self.exchange:
            if hasattr(self.exchange, "close") and callable(self.exchange.close):
                logger.info(f"{Fore.CYAN}# Banishing CCXT Bybit client connection...{Style.RESET_ALL}")
                try:
                    await self.exchange.close()
                    logger.info(f"{Fore.GREEN}CCXT connection banished.{Style.RESET_ALL}")
                except Exception as e:
                    logger.error(f"{Fore.RED}Error banishing CCXT connection: {e}{Style.RESET_ALL}", exc_info=False)
            else:
                logger.debug("Exchange object does not have a close method or is not callable.")
        self.exchange = None
        self._initialized = False
        self._markets_loaded = False

    def _check_initialized(self) -> None:
        """Raises a ConnectionError if the client is not initialized."""
        if not self.exchange or not self._initialized or not self._markets_loaded:
            raise ConnectionError("CCXT client is not initialized, failed to initialize, or markets not loaded.")

    async def fetch_balance(self) -> dict:
        """Fetches account balance (USDT Futures) for V5 Unified/Contract accounts."""
        self._check_initialized()
        logger.debug(f"{Fore.BLUE}# Fetching arcane balance energies (CONTRACT account)...{Style.RESET_ALL}")
        try:
            # Bybit V5 Unified/Contract requires accountType param.
            # CCXT might handle this if defaultType is 'swap', but explicitly setting is safer.
            # For USDT perpetuals, 'CONTRACT' is the account type.
            params = {"accountType": "CONTRACT"}
            balance = await self.exchange.fetch_balance(params=params)
            logger.debug(f"Raw balance data received (Account Type: CONTRACT): {balance}")

            # CCXT aims for standard structure: {'USDT': {'free': ..., 'used': ..., 'total': ...}}
            # Check standard structure first
            if "USDT" in balance:
                usdt_balance = balance["USDT"]
                # Ensure info is present for detailed PNL extraction later if needed
                usdt_balance["info"] = balance.get("info", {})  # Copy top-level info if available
                # Try to populate PNL fields if missing in standard keys but present in raw 'info'
                if "info" in usdt_balance and usdt_balance["info"]:
                    raw_list = usdt_balance["info"].get("result", {}).get("list", [])
                    if raw_list and isinstance(raw_list[0], dict):
                        asset_info = raw_list[0]  # Assuming first item is USDT for CONTRACT account
                        usdt_balance["unrealizedPnl"] = usdt_balance.get("unrealizedPnl") or asset_info.get(
                            "unrealisedPnl"
                        )
                        usdt_balance["cumRealisedPnl"] = usdt_balance.get("cumRealisedPnl") or asset_info.get(
                            "cumRealisedPnl"
                        )
                logger.debug(f"Found standardized USDT balance: {usdt_balance}")
                return usdt_balance

            # If 'USDT' key missing, parse Bybit's V5 structure from 'info'
            # This path handles the 'accountType only support UNIFIED' error case if CCXT didn't parse fully.
            if "info" in balance and "result" in balance["info"] and "list" in balance["info"]["result"]:
                for asset_info in balance["info"]["result"]["list"]:
                    # Find the USDT entry specifically for the CONTRACT account
                    if asset_info.get("accountType") == "CONTRACT" and asset_info.get("coin") == "USDT":
                        logger.debug(f"Parsing V5 balance structure from info: {asset_info}")
                        # Map Bybit keys to CCXT-like standard keys
                        parsed_balance = {
                            "free": float(
                                asset_info.get("availableToWithdraw", 0)
                            ),  # Or 'availableBalance'? Check API docs.
                            "used": float(asset_info.get("usedMargin", 0)),
                            "total": float(asset_info.get("equity", 0)),
                            "unrealizedPnl": float(asset_info.get("unrealisedPnl", 0)),
                            "cumRealisedPnl": float(asset_info.get("cumRealisedPnl", 0)),
                            "info": asset_info,  # Keep raw info
                        }
                        logger.debug(f"Parsed V5 balance for USDT: {parsed_balance}")
                        return parsed_balance

            logger.warning(
                f"{Fore.YELLOW}Could not find standardized USDT CONTRACT balance. Raw response keys: {list(balance.keys())}{Style.RESET_ALL}"
            )
            # Return the 'total' part if available, or an empty dict if completely unparsable
            return balance.get("total", {}) if isinstance(balance.get("total"), dict) else {}

        except ccxt.BadRequest as e:
            # Catch the specific error observed: {"retCode":10001,"retMsg":"accountType only support UNIFIED."...}
            # This might indicate the user has a Unified account but the key/permissions or the call structure needs adjustment.
            # Or potentially the `accountType` param needs to be 'UNIFIED' depending on the exact API setup.
            # Let's log clearly and re-raise. The fix above (params={'accountType': 'CONTRACT'}) should address this for Contract accounts.
            # If this error persists, the user might need a Unified account key or adjust permissions.
            logger.error(
                f"{Fore.RED}CCXT BadRequest fetching balance (likely V5 account type issue): {e}. Ensure API key is for CONTRACT or UNIFIED account and has permissions. Tried with accountType=CONTRACT.{Style.RESET_ALL}",
                exc_info=True,
            )
            raise  # Re-raise specific CCXT error
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(f"{Fore.RED}CCXT Error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
            raise  # Re-raise specific CCXT error
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
            raise  # Re-raise general error

    async def place_order(
        self, symbol: str, side: str, order_type: str, amount: float, price: float | None = None, params: dict = None
    ) -> dict:
        """Places an order using CCXT, handling common exceptions and V5 params."""
        if params is None:
            params = {}
        self._check_initialized()

        # Add V5 category param if not already present
        if "category" not in params:
            params["category"] = "linear"  # For USDT perpetuals

        logger.info(
            f"{Fore.CYAN}# Weaving {order_type.capitalize()} {side.upper()} order: {amount} {symbol} @ {price if price else 'Market'} with params: {params}{Style.RESET_ALL}"
        )
        try:
            # CCXT standard params: stopLossPrice, takeProfitPrice, trailingPercent, reduceOnly
            # These should map to Bybit's 'stopLoss', 'takeProfit', 'trailingStop', 'reduceOnly' fields via CCXT.

            # Bybit V5 specific params might need explicit setting if CCXT mapping is insufficient or for more control:
            # e.g., 'slTriggerBy', 'tpTriggerBy' ('MarkPrice', 'LastPrice', 'IndexPrice')
            # e.g., 'tpslMode' ('Full' or 'Partial') - affects if TP/SL applies to whole position or just this order
            # e.g., 'slOrderType', 'tpOrderType' ('Market' or 'Limit')
            # Add these to the `params` dict if needed based on user input or defaults.

            # Example: If user chose trigger types:
            # if 'stopLossPrice' in params: params['slTriggerBy'] = user_sl_trigger_choice
            # if 'takeProfitPrice' in params: params['tpTriggerBy'] = user_tp_trigger_choice

            # Trailing Stop: CCXT's 'trailingPercent' *should* work. Bybit V5 expects 'trailingStop' as a string percentage.
            # If 'trailingPercent' fails, manually construct:
            # if 'trailingPercent' in params and params['trailingPercent'] > 0:
            #     params['trailingStop'] = str(params['trailingPercent']) # Bybit expects string
            #     # Optionally add activation price if collected from user
            #     # if 'trailingActivationPrice' in params:
            #     #     params['activePrice'] = str(params['trailingActivationPrice'])
            #     # del params['trailingPercent'] # Remove standard key if using specific key

            order = await self.exchange.create_order(symbol, order_type, side, amount, price, params)
            logger.info(f"{Fore.GREEN}Order weaving successful: ID {order.get('id')}{Style.RESET_ALL}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"{Fore.RED}Order weaving failed: Insufficient magical essence (funds). {e}{Style.RESET_ALL}")
            raise
        except ccxt.InvalidOrder as e:
            # Provide more context if possible from the error message
            err_msg = str(e).lower()  # Lowercase for easier matching
            if "order quantity" in err_msg or "order qty" in err_msg or "size" in err_msg:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Invalid quantity. Check min/max order size and step size for {symbol}. {e}{Style.RESET_ALL}"
                )
            elif "order price" in err_msg or "price" in err_msg:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Invalid price. Check price limits, tick size, or market conditions for {order_type} orders. {e}{Style.RESET_ALL}"
                )
            elif "margin" in err_msg:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Margin issue. Check available margin and leverage. {e}{Style.RESET_ALL}"
                )
            elif "reduce-only" in err_msg or "reduceonly" in err_msg:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Reduce-only conflict. Cannot increase position size with reduce-only order. {e}{Style.RESET_ALL}"
                )
            elif "trigger" in err_msg or "stop loss" in err_msg or "take profit" in err_msg:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Invalid TP/SL parameters. Check trigger prices relative to current price/side and trigger type. {e}{Style.RESET_ALL}"
                )
            elif "trailing stop" in err_msg or "trailingstop" in err_msg:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Invalid Trailing Stop parameters. Check percentage/activation price. {e}{Style.RESET_ALL}"
                )
            else:
                logger.error(
                    f"{Fore.RED}Order weaving failed: Flawed incantation (invalid order parameters). {e}{Style.RESET_ALL}"
                )
            raise
        except ccxt.BadSymbol as e:
            logger.error(f"{Fore.RED}Order weaving failed: Invalid symbol sigil '{symbol}'. {e}{Style.RESET_ALL}")
            raise  # Re-raise BadSymbol specifically
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(f"{Fore.RED}CCXT Exchange/Network Anomaly placing order: {e}{Style.RESET_ALL}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected vortex placing order: {e}{Style.RESET_ALL}", exc_info=True)
            raise

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int, since: int | None = None
    ) -> list[list[int | float]]:
        """Fetches OHLCV data, handling specific errors and V5 params."""
        self._check_initialized()
        logger.debug(f"{Fore.BLUE}# Scrying {limit} OHLCV candles for {symbol} ({timeframe})...{Style.RESET_ALL}")
        try:
            # Bybit V5 requires category=linear for USDT perpetuals
            params = {"category": "linear"}
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=since, params=params
            )
            if not ohlcv:
                logger.warning(
                    f"{Fore.YELLOW}No OHLCV data returned for {symbol} ({timeframe}). May be a new pair or invalid timeframe.{Style.RESET_ALL}"
                )
                return []
            logger.debug(f"Received {len(ohlcv)} OHLCV candles.")
            # Return data as is, CCXT standard format expected: [timestamp, open, high, low, close, volume]
            return ohlcv
        except ccxt.BadSymbol as e:
            logger.error(f"{Fore.RED}Invalid symbol sigil for OHLCV: {symbol}. {e}{Style.RESET_ALL}")
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.ExchangeError as e:
            # Improve timeframe error detection
            if "interval" in str(e).lower() or "candle type" in str(e).lower() or "invalid timeframe" in str(e).lower():
                logger.error(
                    f"{Fore.RED}Invalid or unsupported time-shard '{timeframe}' for {symbol}. Check available timeframes. {e}{Style.RESET_ALL}"
                )
                raise ValueError(f"Invalid timeframe: {timeframe}") from e
            else:
                logger.error(f"{Fore.RED}CCXT Exchange Anomaly fetching OHLCV: {e}{Style.RESET_ALL}", exc_info=True)
                raise
        except ccxt.NetworkError as e:
            logger.error(f"{Fore.RED}CCXT Network Rift fetching OHLCV: {e}{Style.RESET_ALL}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected vortex fetching OHLCV: {e}{Style.RESET_ALL}", exc_info=True)
            raise

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetches the latest ticker information for a symbol using V5 params."""
        self._check_initialized()
        logger.debug(f"{Fore.BLUE}# Fetching current price pulse for {symbol}...{Style.RESET_ALL}")
        try:
            # Bybit V5 requires category=linear
            params = {"category": "linear"}
            ticker = await self.exchange.fetch_ticker(symbol, params=params)
            return ticker
        except ccxt.BadSymbol as e:
            logger.error(f"{Fore.RED}Invalid symbol sigil for ticker: {symbol}. {e}{Style.RESET_ALL}")
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Fore.RED}CCXT Exchange/Network Anomaly fetching ticker: {e}{Style.RESET_ALL}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected vortex fetching ticker: {e}{Style.RESET_ALL}", exc_info=True)
            raise

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetches open orders using V5 params, optionally filtered by symbol."""
        self._check_initialized()
        action = f"for {symbol}" if symbol else "all symbols"
        logger.debug(f"{Fore.BLUE}# Fetching active order scrolls {action}...{Style.RESET_ALL}")
        try:
            # Bybit V5 requires 'category': 'linear' for USDT perpetuals
            params = {"category": "linear"}
            # CCXT fetch_open_orders usually handles the symbol filtering internally
            open_orders = await self.exchange.fetch_open_orders(symbol=symbol, params=params)
            return open_orders
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Fore.RED}CCXT Exchange/Network Anomaly fetching open orders: {e}{Style.RESET_ALL}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected vortex fetching open orders: {e}{Style.RESET_ALL}", exc_info=True)
            raise

    async def fetch_positions(self, symbol: str | None = None) -> list[dict]:
        """Fetches open positions using V5 params, optionally filtered by symbol, and filters zero-size."""
        self._check_initialized()
        # CCXT standard fetch_positions takes a list of symbols or None for all
        symbols_list = [symbol] if symbol else None
        action = f"for {symbol}" if symbol else "all symbols"
        logger.debug(f"{Fore.BLUE}# Fetching open position essences {action}...{Style.RESET_ALL}")
        try:
            # Bybit V5 requires 'category': 'linear'
            params = {"category": "linear"}
            # Fetch positions for the specified symbols (or all if symbols_list is None)
            positions = await self.exchange.fetch_positions(symbols=symbols_list, params=params)

            # Filter out zero-sized positions (CCXT standard often includes them)
            open_positions = []
            for p in positions:
                # Determine size: Check standard 'contracts', fallback 'contractSize', then Bybit V5 'info.size'
                size_str = None
                if p.get("contracts") is not None:
                    size_str = str(p["contracts"])
                elif p.get("contractSize") is not None:
                    size_str = str(p["contractSize"])
                elif "info" in p and p["info"] and p["info"].get("size") is not None:
                    size_str = str(p["info"]["size"])  # Bybit V5 uses 'size' in info

                try:
                    # Ensure size is treated as a number for comparison, handle potential None or empty string
                    if size_str is not None and size_str != "" and float(size_str) != 0:
                        open_positions.append(p)
                    # Log positions with zero or invalid size for debugging if needed
                    # else:
                    #    logger.debug(f"Filtered out position for {p.get('symbol')} with size: '{size_str}'")
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not parse position size '{size_str}' for symbol {p.get('symbol')}. Skipping."
                    )

            logger.debug(f"Found {len(open_positions)} open (non-zero size) positions matching filter.")
            return open_positions
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{Fore.RED}CCXT Exchange/Network Anomaly fetching positions: {e}{Style.RESET_ALL}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected vortex fetching positions: {e}{Style.RESET_ALL}", exc_info=True)
            raise


# --- Technical Analysis ---


class TechnicalAnalysis:
    """Performs technical analysis on market data."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.indicator_periods = config.get("indicator_periods", {})
        if ta is None:
            logger.warning(
                f"{Fore.YELLOW}# pandas_ta not found. Technical Analysis Oracle is dormant.{Style.RESET_ALL}"
            )
        else:
            logger.info(f"{Fore.CYAN}# Technical Analysis Oracle initialized.{Style.RESET_ALL}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates technical indicators using pandas_ta based on config."""
        if ta is None:
            logger.error(f"{Fore.RED}Cannot calculate indicators: pandas_ta library is not installed.{Style.RESET_ALL}")
            return df  # Return original dataframe

        if df.empty:
            logger.warning(f"{Fore.YELLOW}Cannot conjure indicators from empty DataFrame.{Style.RESET_ALL}")
            return df

        # Ensure standard column names (lowercase)
        df_in = df.copy()  # Work on a copy
        df_in.columns = [str(col).lower() for col in df_in.columns]
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_in.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_in.columns]
            logger.error(
                f"{Fore.RED}DataFrame lacks required columns for TA: {missing}. Available: {list(df_in.columns)}{Style.RESET_ALL}"
            )
            # Return original df as we cannot proceed
            return df

        # Drop rows with NaN in essential columns before calculating TA
        initial_len = len(df_in)
        df_in.dropna(
            subset=["open", "high", "low", "close"], inplace=True
        )  # Volume NaN might be acceptable for some indicators
        if len(df_in) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df_in)} rows with NaN OHLC data before TA calculation.")
        if df_in.empty:
            logger.error(
                f"{Fore.RED}DataFrame is empty after removing NaN OHLC data. Cannot calculate TA.{Style.RESET_ALL}"
            )
            return df  # Return original

        logger.debug(f"{Fore.BLUE}# Calculating arcane indicators on {len(df_in)} rows...{Style.RESET_ALL}")
        df_out = df_in.copy()

        # Safely get lengths from config, providing defaults if missing
        periods = self.indicator_periods
        sma_short_len = periods.get("sma_short", 20)
        sma_long_len = periods.get("sma_long", 50)
        ema_len = periods.get("ema", 20)
        rsi_len = periods.get("rsi", 14)
        bb_len = periods.get("bbands", 20)
        bb_std = periods.get("bbands_std", 2.0)
        stoch_k = periods.get("stoch_k", 14)
        stoch_d = periods.get("stoch_d", 3)
        stoch_smooth_k = periods.get("stoch_smooth_k", 3)
        atr_len = periods.get("atr", 14)
        # MACD defaults (12, 26, 9) are standard

        try:
            # Define the strategy using pandas_ta
            # Ensure lengths are integers/floats as needed
            custom_strategy = ta.Strategy(
                name="Pyrmethus TA Set",
                ta=[
                    {"kind": "sma", "length": int(sma_short_len)},
                    {"kind": "sma", "length": int(sma_long_len)},
                    {"kind": "ema", "length": int(ema_len)},
                    {"kind": "rsi", "length": int(rsi_len)},
                    {"kind": "macd"},  # Uses default lengths (12, 26, 9)
                    {"kind": "bbands", "length": int(bb_len), "std": float(bb_std)},
                    {"kind": "stoch", "k": int(stoch_k), "d": int(stoch_d), "smooth_k": int(stoch_smooth_k)},
                    {"kind": "atr", "length": int(atr_len)},
                ],
            )
            # Apply the strategy (operates inplace on df_out.ta accessor)
            df_out.ta.strategy(custom_strategy)

            # Check if columns were actually added (strategy might fail silently or add nothing if data too short)
            added_cols = list(set(df_out.columns) - set(df_in.columns))
            if not added_cols:
                # Check length vs required lookback
                min_rows_needed = max(
                    int(sma_long_len), int(bb_len), 26 + 9, int(stoch_k) + int(stoch_d) + int(stoch_smooth_k)
                )  # Estimate
                if len(df_out) < min_rows_needed:
                    logger.warning(
                        f"{Fore.YELLOW}Pandas TA strategy ran but added no new indicator columns. Insufficient data length ({len(df_out)} rows) for lookback periods (e.g., ~{min_rows_needed} needed).{Style.RESET_ALL}"
                    )
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Pandas TA strategy ran but added no new indicator columns. Check configuration or library behavior.{Style.RESET_ALL}"
                    )
            else:
                logger.debug(
                    f"{Fore.GREEN}Successfully calculated indicators. Columns added: {added_cols}{Style.RESET_ALL}"
                )
                # Round indicator values for cleaner display (only newly added columns)
                df_out[added_cols] = df_out[added_cols].round(5)

            # Reindex df_out to match the original df's index to include rows that had NaNs initially
            # Fill potentially missing indicator values in those re-introduced rows with NaN
            df_final = df_out.reindex(df.index)
            return df_final

        except AttributeError as e:
            if "'DataFrame' object has no attribute 'ta'" in str(e):
                logger.error(
                    f"{Fore.RED}Pandas TA extension not found or failed to attach. Is pandas_ta installed and imported correctly?{Style.RESET_ALL}",
                    exc_info=True,
                )
            else:
                logger.error(f"{Fore.RED}Attribute error calculating indicators: {e}{Style.RESET_ALL}", exc_info=True)
            return df  # Return original df on error
        except Exception as e:
            logger.error(f"{Fore.RED}Error calculating indicators: {e}{Style.RESET_ALL}", exc_info=True)
            return df  # Return original df on error

    def calculate_pivot_points(self, df_period: pd.DataFrame) -> dict | None:
        """Calculates Classic Pivot Points based on the provided DataFrame (expects 1 row: last completed HLC)."""
        if df_period is None or df_period.empty:
            logger.warning(
                f"{Fore.YELLOW}Insufficient data for pivot point calculation (DataFrame empty).{Style.RESET_ALL}"
            )
            return None

        # Ensure columns are lowercase
        df_period.columns = [str(col).lower() for col in df_period.columns]
        required_cols = ["high", "low", "close"]
        if not all(col in df_period.columns for col in required_cols):
            logger.error(
                f"{Fore.RED}Pivot DataFrame missing required columns: {required_cols}. Available: {list(df_period.columns)}{Style.RESET_ALL}"
            )
            return None

        # Use the first row provided, assuming it's the last *completed* period's data
        if len(df_period) < 1:
            logger.warning(f"{Fore.YELLOW}Insufficient rows in pivot DataFrame.{Style.RESET_ALL}")
            return None

        try:
            # Use the first row (index 0) passed to this function
            last_period = df_period.iloc[0]
            high = last_period["high"]
            low = last_period["low"]
            close = last_period["close"]

            # Check for NaN values before calculation
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                candle_time = last_period.name if last_period.name else "Unknown Time"
                logger.warning(
                    f"{Fore.YELLOW}NaN values encountered in HLC data for pivot calculation (H:{high}, L:{low}, C:{close}) for candle '{candle_time}'. Cannot calculate pivots.{Style.RESET_ALL}"
                )
                return None

            # Ensure values are floats for calculation
            high, low, close = float(high), float(low), float(close)

            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)

            candle_time_str = (
                last_period.name.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(last_period.name, pd.Timestamp)
                else str(last_period.name)
            )
            logger.debug(
                f"{Fore.BLUE}# Pivot points calculated based on H={high}, L={low}, C={close} from period ending {candle_time_str}{Style.RESET_ALL}"
            )
            return {
                "Resistance 3 (R3)": r3,
                "Resistance 2 (R2)": r2,
                "Resistance 1 (R1)": r1,
                "Pivot (P)": pivot,
                "Support 1 (S1)": s1,
                "Support 2 (S2)": s2,
                "Support 3 (S3)": s3,
            }
        except KeyError as e:
            logger.error(f"{Fore.RED}Missing HLC columns for pivot calculation: {e}{Style.RESET_ALL}")
            return None
        except (ValueError, TypeError) as e:
            logger.error(
                f"{Fore.RED}Data type error during pivot calculation: {e}. Ensure HLC are numeric.{Style.RESET_ALL}"
            )
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Error calculating pivot points: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def determine_trend(self, df: pd.DataFrame) -> str:
        """Determines a simple trend based on short vs long SMA crossover."""
        if ta is None:
            return f"{Style.DIM}Unknown (pandas_ta missing){Style.RESET_ALL}"
        # Need at least 2 non-NaN values for comparison
        if df.empty:
            return f"{Style.DIM}Unknown (No Data){Style.RESET_ALL}"

        # Get column names based on config lengths (expecting pandas_ta default naming)
        sma_short_len = self.indicator_periods.get("sma_short", 20)
        sma_long_len = self.indicator_periods.get("sma_long", 50)
        # pandas_ta names columns like 'SMA_20', 'SMA_50' (uppercase) or potentially 'sma_20'
        # Check for both cases, prefer lowercase if calculate_indicators forces it
        sma_short_col_options = [f"sma_{sma_short_len}", f"SMA_{sma_short_len}"]
        sma_long_col_options = [f"sma_{sma_long_len}", f"SMA_{sma_long_len}"]

        # Find which column names actually exist in the DataFrame
        sma_short_col = next((col for col in sma_short_col_options if col in df.columns), None)
        sma_long_col = next((col for col in sma_long_col_options if col in df.columns), None)

        if not sma_short_col or not sma_long_col:
            return f"{Style.DIM}Unknown (Missing SMAs){Style.RESET_ALL}"

        # Get the last two rows where BOTH SMAs are not NaN
        valid_sma_df = df[[sma_short_col, sma_long_col]].dropna()
        if len(valid_sma_df) < 2:
            return f"{Style.DIM}Unknown (Insufficient SMA Data){Style.RESET_ALL}"

        last_row = valid_sma_df.iloc[-1]
        prev_row = valid_sma_df.iloc[-2]

        trend = "Neutral / Sideways"
        color = Style.DIM  # Default neutral color

        # Current state
        is_bullish_now = last_row[sma_short_col] > last_row[sma_long_col]
        is_bearish_now = last_row[sma_short_col] < last_row[sma_long_col]

        # Previous state
        was_bullish_prev = prev_row[sma_short_col] > prev_row[sma_long_col]

        if is_bullish_now:
            trend = "Up"
            color = Fore.GREEN
            # Check for crossover from non-bullish (bearish or equal) in the previous valid period
            if not was_bullish_prev:
                trend += " (Bullish Crossover)"
                color = Fore.GREEN + Style.BRIGHT
        elif is_bearish_now:
            trend = "Down"
            color = Fore.RED
            # Check for crossover from bullish in the previous valid period
            if was_bullish_prev:
                trend += " (Bearish Crossover)"
                color = Fore.RED + Style.BRIGHT

        logger.debug(f"{Fore.BLUE}# Trend determined: {trend}{Style.RESET_ALL}")
        return f"{color}{trend}{Style.RESET_ALL}"

    def generate_ta_summary(self, df: pd.DataFrame) -> dict[str, str]:
        """Generates a concise summary of key TA indicators from the last valid row."""
        summary = {}
        if ta is None:
            return {"Status": f"{Fore.YELLOW}pandas_ta missing{Style.RESET_ALL}"}
        if df.empty:
            return {"Status": f"{Fore.YELLOW}No data for summary{Style.RESET_ALL}"}

        # Find the last row with at least a 'close' value
        last_valid_row_idx = df["close"].last_valid_index()
        if last_valid_row_idx is None:
            return {"Status": f"{Fore.YELLOW}No valid data points{Style.RESET_ALL}"}
        last = df.loc[last_valid_row_idx]

        # Standardize column access (assume lowercase from calculate_indicators or check both cases)
        df_cols = df.columns.tolist()

        # Helper to safely get value from the last valid row, checking potential column names
        def get_val(col_name_options: list[str], default="N/A"):
            for col_name in col_name_options:
                if col_name in df_cols and pd.notna(last.get(col_name)):
                    return last.get(col_name)
            return default

        # Price (use lowercase 'close')
        price = get_val(["close"])
        summary["Last Price"] = (
            f"{Fore.YELLOW}{price:.4f}{Style.RESET_ALL}" if isinstance(price, (float, int, np.number)) else price
        )

        # Trend (pass the full df for history check)
        summary["Trend (SMA)"] = self.determine_trend(df)

        # RSI (e.g., RSI_14 or rsi_14)
        rsi_len = self.indicator_periods.get("rsi", 14)
        rsi_cols = [f"rsi_{rsi_len}", f"RSI_{rsi_len}"]
        rsi = get_val(rsi_cols)
        if isinstance(rsi, (float, int, np.number)):
            rsi_val = f"{rsi:.2f}"
            if rsi > 70:
                summary["RSI"] = f"{Fore.RED}{rsi_val} (Overbought){Style.RESET_ALL}"
            elif rsi < 30:
                summary["RSI"] = f"{Fore.GREEN}{rsi_val} (Oversold){Style.RESET_ALL}"
            else:
                summary["RSI"] = f"{Fore.WHITE}{rsi_val} (Neutral){Style.RESET_ALL}"
        else:
            summary["RSI"] = rsi  # Keep "N/A" or other non-numeric value

        # MACD (pandas_ta defaults: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9 or lowercase)
        hist = get_val(["macdh_12_26_9", "MACDh_12_26_9"])
        if isinstance(hist, (float, int, np.number)):
            hist_val = f"{hist:.4f}"
            if hist > 0:
                summary["MACD Hist"] = f"{Fore.GREEN}{hist_val} (Bullish){Style.RESET_ALL}"
            elif hist < 0:
                summary["MACD Hist"] = f"{Fore.RED}{hist_val} (Bearish){Style.RESET_ALL}"
            else:
                summary["MACD Hist"] = f"{Fore.WHITE}{hist_val} (Neutral){Style.RESET_ALL}"
        else:
            summary["MACD Hist"] = hist

        # Stochastic (e.g., STOCHk_14_3_3, STOCHd_14_3_3 or lowercase)
        stoch_k_len = self.indicator_periods.get("stoch_k", 14)
        stoch_d_len = self.indicator_periods.get("stoch_d", 3)
        stoch_smooth_k_len = self.indicator_periods.get("stoch_smooth_k", 3)
        stoch_k_cols = [
            f"stochk_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}",
            f"STOCHk_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}",
        ]
        stoch_d_cols = [
            f"stochd_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}",
            f"STOCHd_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}",
        ]
        k = get_val(stoch_k_cols)
        d = get_val(stoch_d_cols)
        if isinstance(k, (float, int, np.number)) and isinstance(d, (float, int, np.number)):
            stoch_val = f"K:{k:.2f}, D:{d:.2f}"
            if k > 80 and d > 80:
                summary["Stochastic"] = f"{Fore.RED}{stoch_val} (Overbought){Style.RESET_ALL}"
            elif k < 20 and d < 20:
                summary["Stochastic"] = f"{Fore.GREEN}{stoch_val} (Oversold){Style.RESET_ALL}"
            else:
                summary["Stochastic"] = f"{Fore.WHITE}{stoch_val}{Style.RESET_ALL}"
        else:
            summary["Stochastic"] = "N/A"  # Combine if either k or d is missing

        # ATR (Volatility - e.g., ATR_14 or atr_14)
        atr_len = self.indicator_periods.get("atr", 14)
        atr_cols = [f"atr_{atr_len}", f"ATR_{atr_len}"]
        atr = get_val(atr_cols)
        if isinstance(atr, (float, int, np.number)):
            summary["ATR"] = f"{Fore.MAGENTA}{atr:.4f}{Style.RESET_ALL}"
        else:
            summary["ATR"] = atr

        return summary


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
        except OSError:
            term_width = 80  # Default width
        f"{self.colors['primary']}{'=' * term_width}{self.colors['reset']}"

    def display_menu(self, title: str, options: list[str], prompt: str = "Choose your action") -> str:
        """Displays a menu and gets user choice. Handles EOFError/KeyboardInterrupt.
        This is a BLOCKING function, intended to be run in an executor thread.
        """
        self.display_header(title)
        for _i, _option in enumerate(options, 1):
            # Option text might already contain color codes (e.g., for disabled items)
            pass  # Reset after each option

        while True:
            try:
                choice = input(f"\n{self.colors['input_prompt']}{prompt} (1-{len(options)}): {self.colors['reset']}")
                choice = choice.strip()
                if choice.isdigit() and 1 <= int(choice) <= len(options):
                    logger.debug(f"Menu '{title}' choice: {choice}")
                    return choice
                else:
                    # Use print_warning for invalid input feedback (avoids calling another blocking func)
                    pass
            except EOFError:
                logger.warning("EOF detected in display_menu, signalling exit request.")
                # Raise EOFError to be caught by the async executor wrapper
                raise EOFError("EOF detected during menu input.")
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt caught during menu input, signalling exit request.")
                # Raise KeyboardInterrupt to be caught by the async executor wrapper
                raise KeyboardInterrupt("KeyboardInterrupt during menu input.")

    def get_input(
        self,
        prompt: str,
        default: str | None = None,
        required: bool = True,
        input_type: type = str,
        validation_func: Callable | None = None,
    ) -> Any:
        """Gets validated user input. Handles EOFError/KeyboardInterrupt, type conversion, and validation.
        This is a BLOCKING function, intended to be run in an executor thread.
        """
        while True:
            prompt_full = f"{self.colors['input_prompt']}{prompt}"
            if default is not None:
                prompt_full += f" [{self.colors['accent']}default: {default}{self.colors['input_prompt']}]"
            prompt_full += f": {self.colors['reset']}"
            try:
                user_input = input(prompt_full).strip()
            except EOFError:
                logger.warning(f"EOF detected getting input for '{prompt}'.")
                raise EOFError("Input stream closed unexpectedly.")  # Propagate EOF
            except KeyboardInterrupt:
                logger.warning(f"KeyboardInterrupt caught getting input for '{prompt}'.")
                raise KeyboardInterrupt  # Propagate interrupt

            if not user_input and default is not None:
                user_input = str(default)  # Use default value
                logger.debug(f"Input for '{prompt}' using default: {default}")

            if required and not user_input:
                # Use print_error directly here as it's synchronous
                continue

            if not user_input and not required:
                logger.debug(f"Optional input for '{prompt}' left blank.")
                return None  # Return None for blank optional input

            # Proceed with conversion and validation even if default was used
            try:
                value: Any
                if input_type == bool:
                    if user_input.lower() in ("true", "1", "t", "y", "yes"):
                        value = True
                    elif user_input.lower() in ("false", "0", "f", "n", "no"):
                        value = False
                    else:
                        raise ValueError("Invalid boolean value. Use True/False, Yes/No, 1/0.")
                elif input_type == str:
                    value = user_input  # Already a string
                else:
                    # Attempt conversion to the target type (e.g., float, int)
                    value = input_type(user_input)

                # Perform validation if a function is provided
                if validation_func:
                    validation_result = validation_func(value)  # Pass the *converted* value
                    if (
                        isinstance(validation_result, str) or validation_result is False
                    ):  # Validation func returned an error message
                        continue
                    # If validation_result is True or None (implicit success), proceed

                logger.debug(f"Input for '{prompt}': {repr(value)} (type: {input_type})")
                return value
            except ValueError:
                # Provide more specific feedback for common float errors
                if input_type == float:
                    pass
                else:
                    pass
            except Exception:  # Catch errors during validation_func execution
                pass

    def print_error(self, message: str) -> None:
        """Prints an error message."""
        # Avoid logging redundant "Error:" prefix if logger already adds levelname
        logger.error(message)

    def print_success(self, message: str) -> None:
        """Prints a success message."""
        logger.info(message)  # Log success messages too

    def print_warning(self, message: str) -> None:
        """Prints a warning message."""
        logger.warning(message)

    def print_info(self, message: str) -> None:
        """Prints an informational message."""
        # Optionally log info messages if needed for debugging flow
        # logger.info(message)

    def _contains_ansi_codes(self, text: str) -> bool:
        """Checks if a string contains ANSI escape codes."""
        # Simple check for the escape character followed by [
        return "\x1b[" in text

    def print_table(
        self,
        data: pd.DataFrame | list[dict] | dict,
        title: str | None = None,
        float_format: str = "{:.4f}",
        index: bool = False,
    ) -> None:
        """Prints data in a formatted table using tabulate if available, with better handling for colored strings."""
        if title:
            pass

        if data is None or (isinstance(data, (list, dict, pd.DataFrame)) and not data):
            return

        try:
            df = None
            headers = "keys"  # Default for dicts in tabulate
            table_data = None
            contains_color = False  # Flag to detect colored strings

            # --- Prepare Data for Tabulate ---
            if isinstance(data, pd.DataFrame):
                df = data
                if df.empty:
                    return
                # Check for colored strings in DataFrame cells
                if df.map(lambda x: isinstance(x, str) and self._contains_ansi_codes(x)).any().any():
                    contains_color = True
                # Prepare DataFrame for tabulate
                headers = list(df.columns)
                if index:
                    df_to_print = df.reset_index()
                    headers.insert(0, str(df_to_print.columns[0]))  # Add index name to headers
                else:
                    df_to_print = df
                table_data = df_to_print.values.tolist()

            elif isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
                # Convert list of dicts to format suitable for tabulate
                headers = list(data[0].keys())  # Use keys from the first dict
                table_data = []
                for item in data:
                    row = []
                    for h in headers:
                        value = item.get(h)
                        if isinstance(value, str) and self._contains_ansi_codes(value):
                            contains_color = True
                        row.append(value)
                    table_data.append(row)
                if contains_color:
                    pass

            elif isinstance(data, dict) and data:
                # Format dictionary as a two-column table (Key, Value)
                headers = ["Parameter", "Value"]
                table_data = []
                for key, value in data.items():
                    value_str = str(value)  # Default string representation
                    # Check for color codes BEFORE formatting numbers
                    if isinstance(value, str) and self._contains_ansi_codes(value):
                        contains_color = True
                        # Keep the colored string as is for direct printing later
                        table_data.append([str(key), value_str])
                        continue  # Skip number formatting for colored strings

                    # Apply float formatting if applicable
                    if isinstance(value, (float, np.floating)):
                        try:
                            value_str = float_format.format(value)
                        except (ValueError, TypeError):
                            pass  # Keep original string if format fails

                    table_data.append([str(key), value_str])

                # If dict contained color, print manually as tabulate struggles with embedded codes
                if contains_color:
                    max(len(str(k)) for k in data) if data else 0
                    for _k, _v in data.items():
                        # Align keys, print value (which might be colored)
                        pass  # v already contains color/reset
                    return  # Stop here for colored dicts

            else:
                # Handle cases like empty list/dict or unsupported types
                if not data:
                    pass
                else:
                    self.print_error(f"Unsupported data type for print_table: {type(data)}")
                return

            # --- Use Tabulate or Fallback ---
            if tabulate and table_data and not contains_color:  # Only use tabulate if no color codes detected
                # Define float formatting function for tabulate
                def fmt_num(val):
                    if isinstance(val, (float, np.floating)):
                        try:
                            return float_format.format(val)
                        except (ValueError, TypeError):
                            return str(val)
                    # Handle None explicitly for tabulate
                    return "" if val is None else val

                [[fmt_num(item) for item in row] for row in table_data]

            elif df is not None:  # Fallback for Pandas DataFrame if tabulate is missing OR contains color
                if not tabulate:
                    pass
                try:
                    term_width = os.get_terminal_size().columns
                except OSError:
                    term_width = 120  # Wider default fallback
                pd.set_option("display.max_rows", 100)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", term_width)
                pd.set_option("display.expand_frame_repr", True)
                # Simplified float format lambda
                pd.options.display.float_format = (
                    float_format.format if not contains_color else None
                )  # Don't format if color exists
                pd.reset_option("display.float_format")  # Reset option

            elif table_data:  # Fallback for list/dict if tabulate missing OR contains color
                if not tabulate:
                    pass
                # Basic fallback print (less aligned, handles color better than bad tabulate)
                if isinstance(headers, list):
                    pass
                for row in table_data:
                    # Format numbers if no color, otherwise print raw
                    formatted_row = []
                    for item in row:
                        if contains_color:
                            formatted_row.append(str(item))  # Keep raw string with color
                        else:
                            # Apply float format if applicable
                            if isinstance(item, (float, np.floating)):
                                try:
                                    formatted_row.append(float_format.format(item))
                                except (ValueError, TypeError):
                                    formatted_row.append(str(item))
                            else:
                                formatted_row.append(str("" if item is None else item))

        except Exception as e:
            logger.error(f"Error during print_table: {e}", exc_info=True)
            self.print_error(f"An error occurred while trying to display the table: {e}")

    def wait_for_enter(self, prompt: str = "Press Enter to continue...") -> None:
        """Pauses execution until Enter is pressed. Handles EOFError/KeyboardInterrupt.
        This is a BLOCKING function, intended to be run in an executor thread.
        """
        try:
            input(f"\n{self.colors['accent']}{prompt}{self.colors['reset']}")
        except EOFError:
            logger.warning("EOF detected waiting for Enter.")
            raise EOFError("EOF detected while waiting for Enter.")
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt caught waiting for Enter.")
            raise KeyboardInterrupt("KeyboardInterrupt while waiting for Enter.")

    def display_chart(self, data: list[float], title: str) -> None:
        """Displays a simple ASCII chart using asciichartpy."""
        if not data:
            self.print_warning(f"No data available for chart: {title}")
            return

        chart_height = self.config_manager.config.get("chart_height", 10)
        try:
            # Clean data: ensure numeric, replace NaN/inf with previous valid value
            plot_data = []
            last_valid = np.nan
            for d in data:
                # Check if it's a number type and finite
                if isinstance(d, (int, float, np.number)) and np.isfinite(d):
                    last_valid = float(d)
                    plot_data.append(last_valid)
                elif not np.isnan(last_valid):  # Use last valid if current is bad/non-numeric
                    plot_data.append(last_valid)
                # If last_valid is still NaN (e.g., at the start), append NaN
                # asciichartpy handles internal NaNs gracefully
                else:
                    plot_data.append(np.nan)

            # Check if any valid points remain after cleaning
            if not any(np.isfinite(d) for d in plot_data):
                self.print_warning(f"No valid numeric data points for chart: {title}")
                return

            # Get the primary color code from colorama Fore object (e.g., '36' for cyan)
            primary_color_code_match = re.search(r"\x1b\[(\d+)m", self.colors["primary"])
            chart_color_code = primary_color_code_match.group(1) if primary_color_code_match else None

            # Map colorama code to asciichartpy color constant if possible
            color_map = {
                "31": asciichart.red,
                "32": asciichart.green,
                "33": asciichart.yellow,
                "34": asciichart.blue,
                "35": asciichart.magenta,
                "36": asciichart.cyan,
                "37": asciichart.white,
            }
            chart_color = color_map.get(chart_color_code, asciichart.default)  # Default if code not found

            # Generate chart using asciichartpy
            # Use default config for asciichartpy unless specific settings are needed
            asciichart.plot(plot_data, {"height": chart_height, "colors": [chart_color]})
            # Print the chart - asciichartpy includes ANSI codes directly

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
        self._shutdown_event = asyncio.Event()
        self._active_tasks: set[asyncio.Task] = set()  # Store tasks created by _create_task
        self._final_exit_code = 0  # Store exit code set by shutdown

    def _create_task(self, coro, name: str | None = None) -> asyncio.Task:
        """Creates an asyncio task, names it (if possible), and tracks it."""
        # Ensure name is provided for better debugging
        task_name = name or f"Task-{id(coro)}"
        task = asyncio.create_task(coro, name=task_name)  # name param not available < 3.8

        self._active_tasks.add(task)
        # Remove task from set when it's done to prevent memory leak
        task.add_done_callback(self._active_tasks.discard)
        logger.debug(f"Task '{task_name}' created.")
        return task

    def _setup_signal_handlers(self) -> None:
        """Sets up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        signals_to_handle = (signal.SIGINT, signal.SIGTERM)

        def signal_handler(sig) -> None:
            """Wrapper function to schedule shutdown asynchronously."""
            logger.info(f"Received signal {sig.name}. Initiating shutdown...")
            # Schedule the shutdown coroutine without blocking the handler
            # Check if shutdown is already in progress
            if not self._shutdown_event.is_set():
                # Use create_task to ensure it runs on the loop and is tracked
                self._create_task(self.shutdown(signal=sig), name=f"ShutdownHandler_{sig.name}")
            else:
                logger.debug(f"Shutdown already in progress, ignoring signal {sig.name}")

        for sig in signals_to_handle:
            try:
                loop.add_signal_handler(sig, partial(signal_handler, sig))
                logger.debug(f"Signal handler set for {sig.name}")
            except NotImplementedError:
                # Common on Windows for SIGTERM/SIGINT in basic Python console
                logger.warning(
                    f"{Fore.YELLOW}Signal handler for {sig.name} not fully supported on this platform (e.g., Windows console). Use Exit menu or Ctrl+C/Ctrl+Break.{Style.RESET_ALL}"
                )
            except ValueError as e:
                logger.warning(
                    f"{Fore.YELLOW}Signal handler for {sig.name} could not be set (may already be handled or loop issue): {e}{Style.RESET_ALL}"
                )

        logger.info(f"{Fore.CYAN}# Signal conduits established for graceful departure.{Style.RESET_ALL}")

    async def setup_credentials(self) -> bool:
        """Loads API credentials from .env file."""
        load_dotenv()  # Load variables from .env file into environment
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        testnet_str = os.getenv("TESTNET", "False")  # Default to 'False' if not set
        # Robust boolean conversion for TESTNET
        testnet = testnet_str.strip().lower() in ("true", "1", "t", "y", "yes")

        # Check if keys are missing or still placeholders
        if not api_key or not api_secret or api_key == "your_api_key_here" or api_secret == "your_api_secret_here":
            logger.error(
                "API credentials (BYBIT_API_KEY, BYBIT_API_SECRET) not found or are placeholders in .env file."
            )
            self.ui.print_error("API credentials not found or not configured in .env file.")
            self.ui.print_warning(
                "Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are correctly set in the .env file."
            )
            self.ui.print_info("Obtain keys from Bybit: Account -> API Management.")
            self.credentials = None
            return False
        else:
            logger.info(
                f"{Fore.GREEN}API credential sigils loaded successfully. Testnet mode: {testnet}{Style.RESET_ALL}"
            )
            self.credentials = APICredentials(api_key.strip(), api_secret.strip(), testnet)
            return True

    async def initialize(self) -> None:
        """Initializes the terminal, including API client connection."""
        self.ui.display_header("Pyrmethus Bybit Terminal - Initialization")
        self.ui.print_info(f"{Fore.CYAN}# Awakening terminal energies...{Style.RESET_ALL}")

        # Setup credentials first
        if not await self.setup_credentials():
            self.ui.print_warning("Running in limited mode without valid API credentials.")
            self.ui.print_info("Authenticated actions (trading, balance, positions, orders) will be unavailable.")
            # Don't wait, proceed to menu with limited options
            return  # Client will remain None

        # Credentials valid, attempt to initialize client
        if not self.credentials:  # Should not happen if setup_credentials returned True, but check anyway
            logger.error("Credentials object is None after successful setup, cannot initialize client.")
            self.ui.print_error("Internal error: Credentials not set.")
            return

        self.exchange_client = BybitFuturesCCXTClient(self.credentials, self.config_manager.config)
        try:
            await self.exchange_client.initialize()
            self.ui.print_success("Exchange connection established and markets loaded.")
            # Proceed to menu
        except ConnectionError as e:
            # Specific connection/auth/market load errors handled during initialize()
            self.ui.print_error(f"Failed to initialize exchange client: {e}")
            self.ui.print_warning(
                "Proceeding without a fully functional exchange client. Authenticated features will fail."
            )
            # Ensure client is None if init failed, even if object was created
            if self.exchange_client:  # Close only if object exists
                await self.exchange_client.close()  # Explicitly close if init failed partially
            self.exchange_client = None
            # Don't wait, proceed to menu
        except Exception as e:
            # Catch unexpected errors during initialization process
            self.ui.print_error(f"Unexpected critical error during initialization: {e}")
            logger.critical("Critical initialization error", exc_info=True)
            # Trigger immediate shutdown if initialization fails critically
            # Use create_task to avoid awaiting shutdown within the exception handler
            self._create_task(self.shutdown(exit_code=1, signal="InitFailure"), name="ShutdownOnInitFailure")

    async def shutdown(self, signal=None, exit_code=0) -> None:
        """Gracefully shuts down the application, cancelling tasks and closing connections."""
        if self._shutdown_event.is_set():  # Prevent concurrent shutdowns
            if signal:
                logger.debug(f"Shutdown already in progress, ignoring signal {getattr(signal, 'name', signal)}")
            return

        self._running = False  # Signal loops to stop FIRST
        self._shutdown_event.set()  # Signal waiters SECOND
        self._final_exit_code = exit_code  # Store exit code

        signal_name = f"signal {getattr(signal, 'name', signal)}" if signal else "request"
        logger.info(f"Shutdown initiated by {signal_name}... Setting exit code to {exit_code}")
        self.ui.print_info(
            f"\n{self.ui.colors['primary']}# Banishing terminal... Please wait.{self.ui.colors['reset']}"
        )

        # Cancel tracked background tasks (excluding the shutdown task itself)
        current_task = asyncio.current_task()
        # Make a copy as the set might change during iteration if tasks complete quickly
        tasks_to_cancel = {task for task in self._active_tasks if task is not current_task}

        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} active background tasks...")
            for task in tasks_to_cancel:
                # Get name before cancelling, as task object might become invalid after
                task_name = task.get_name() if sys.version_info >= (3, 8) else f"Task-{id(task)}"
                logger.debug(f"Cancelling task '{task_name}'...")
                task.cancel()

            # Wait for tasks to finish cancelling (or raise exceptions)
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            cancelled_count = 0
            error_count = 0
            # Iterate through results and original task list (order should match gather)
            for i, result in enumerate(results):
                # Retrieve task reliably even if set changed
                task = list(tasks_to_cancel)[i]
                task_name = task.get_name() if sys.version_info >= (3, 8) else f"Task-{id(task)}"
                if isinstance(result, asyncio.CancelledError):
                    cancelled_count += 1
                    logger.debug(f"Task '{task_name}' cancelled successfully.")
                elif isinstance(result, Exception):
                    error_count += 1
                    # Log the exception associated with the failed task
                    logger.error(
                        f"Error during background task '{task_name}' cancellation/completion: {result}", exc_info=result
                    )
                # Log tasks that completed normally during shutdown too?
                # else: logger.debug(f"Task '{task_name}' completed normally during shutdown.")

            logger.info(f"Background tasks processing complete: {cancelled_count} cancelled, {error_count} errors.")
        else:
            logger.info("No active background tasks to cancel.")

        # Close exchange client connection
        if self.exchange_client:
            logger.info("Closing exchange client connection...")
            await self.exchange_client.close()  # close() is idempotent and handles state

        logger.info(f"Terminal shutdown sequence complete. Final exit code set to: {self._final_exit_code}")
        # The "Terminal has faded" message is now part of the final block in main

    async def run(self) -> None:
        """Runs the main terminal loop, handling menus and actions."""
        self._setup_signal_handlers()
        await self.initialize()

        # Check if shutdown was triggered during initialization (e.g., critical error)
        if self._shutdown_event.is_set():
            logger.warning("Shutdown triggered during initialization, exiting run loop.")
            return  # Don't start the main loop

        self._running = True
        logger.info("Starting main terminal loop.")

        while self._running:
            # Check shutdown flag at the start of each loop iteration
            if self._shutdown_event.is_set():
                logger.info("Shutdown detected at start of loop, breaking.")
                break

            menu_task = None
            shutdown_wait_task = None
            try:
                # Get menu choice asynchronously using the executor helper
                menu_task = self._create_task(self.ui_display_menu_async(), name="MainMenuDisplay")
                # Also create a task to wait for the shutdown event
                shutdown_wait_task = self._create_task(self._shutdown_event.wait(), name="ShutdownWait")

                # Wait for either menu choice or shutdown signal
                done, pending = await asyncio.wait({menu_task, shutdown_wait_task}, return_when=asyncio.FIRST_COMPLETED)

                choice = None
                shutdown_triggered = False

                # Check which task completed
                if shutdown_wait_task in done:
                    shutdown_triggered = True
                    logger.info("Shutdown event triggered completion.")
                    # Menu task might still be running (user hasn't entered choice), needs cancellation
                    if menu_task in pending:
                        logger.debug("Cancelling pending menu task due to shutdown.")
                        menu_task.cancel()
                        # Wait briefly for cancellation to be processed if needed
                        # await asyncio.sleep(0.1)

                if menu_task in done:
                    try:
                        choice = await menu_task  # Get result (the user's choice string) or raise exception
                    except asyncio.CancelledError:
                        logger.info("Menu display task was cancelled, likely by shutdown.")
                        # If shutdown wasn't the trigger, this is unexpected
                        if not shutdown_triggered:
                            logger.warning("Menu task cancelled unexpectedly.")
                            # Consider triggering shutdown if menu fails weirdly? Or just retry loop.
                        shutdown_triggered = True  # Ensure loop breaks if menu cancelled
                    except (EOFError, KeyboardInterrupt) as e:
                        # These are now raised by the UI methods via the async wrapper
                        logger.info(f"'{type(e).__name__}' received from menu, initiating shutdown.")
                        if (
                            not self._shutdown_event.is_set()
                        ):  # Avoid duplicate calls if signal handler already triggered
                            self._create_task(
                                self.shutdown(signal=type(e).__name__, exit_code=1),
                                name=f"ShutdownOn{type(e).__name__}",
                            )
                        shutdown_triggered = True  # Ensure loop breaks
                    except Exception as e:
                        logger.error(f"Error retrieving menu choice: {e}", exc_info=True)
                        self.ui.print_error(f"An error occurred in the menu display/input: {e}")
                        await self.ui_wait_for_enter_async()  # Pause before retrying menu
                        continue  # Go to next loop iteration to retry menu

                # Cancel any remaining pending tasks (e.g., if menu finished, cancel shutdown wait)
                for task in pending:
                    task.cancel()
                    try:
                        await task  # Await cancellation to suppress warnings
                    except asyncio.CancelledError:
                        pass

                # Check again if shutdown happened while waiting or handling results
                if shutdown_triggered or self._shutdown_event.is_set():
                    logger.info("Shutdown detected after wait/menu handling, breaking loop.")
                    self._running = False  # Ensure flag is set
                    break

                # If choice is None at this point, it means shutdown happened or an error occurred
                if choice is None:
                    if not self._running:
                        break  # Break if shutdown flag set
                    logger.debug("Choice is None after wait, continuing loop (likely shutdown or menu error).")
                    continue  # Continue to next loop iteration (will likely break due to shutdown flag)

                # --- Action Mapping ---
                await self.process_menu_choice(choice)

            except Exception as loop_err:
                # Catch unexpected errors within the main loop logic itself
                logger.critical(f"Unexpected critical error in main run loop: {loop_err}", exc_info=True)
                self.ui.print_error(f"A critical error occurred in the main loop: {loop_err}")
                # Trigger shutdown on critical loop errors
                if not self._shutdown_event.is_set():
                    self._create_task(self.shutdown(signal="MainLoopError", exit_code=1), name="ShutdownOnLoopError")
                # Break the loop after attempting shutdown
                self._running = False  # Ensure loop terminates
                break

        logger.info("Main run loop finished.")

    async def process_menu_choice(self, choice: str) -> None:
        """Determines and executes the action based on the menu choice."""
        is_authenticated = self.exchange_client is not None and self.exchange_client._initialized

        # Define all possible menu options: (Display Text, Action Method, Requires Auth)
        # Use None for method if it requires auth but client isn't ready, or for Exit.
        all_menu_options = [
            ("Place Order", self.place_order_menu, True),
            ("View Account Balance", self.view_balance, True),
            ("View Open Orders", self.view_open_orders, True),
            ("View Open Positions", self.view_open_positions, True),
            ("Check Current Price", self.check_price_menu, True),
            (
                "Technical Analysis",
                self.technical_analysis_menu,
                True,
            ),  # Keep TA available even if no auth? Needs fetch_ohlcv
            ("Settings", self.settings_menu, False),  # Settings always available
            ("Exit", None, False),  # Special case handled explicitly
        ]

        # Build the list of *currently available* actions based on auth status
        # This list maps the displayed menu index (1-based) to the correct action/state
        available_actions_map = {}
        current_menu_index = 1
        for text, method, requires_auth in all_menu_options:
            if requires_auth and not is_authenticated:
                # Action is unavailable due to auth, store None for method
                available_actions_map[str(current_menu_index)] = (text, None, True)
            else:
                # Action is available (either doesn't need auth, or auth is present)
                available_actions_map[str(current_menu_index)] = (text, method, requires_auth)
            current_menu_index += 1

        # --- Handle Choice ---
        if choice in available_actions_map:
            option_text, action_method, requires_auth = available_actions_map[choice]

            # Handle Exit explicitly
            if option_text == "Exit":
                if not self._shutdown_event.is_set():  # Avoid duplicate calls
                    logger.info("Exit option selected.")
                    self._create_task(self.shutdown(signal="Exit Menu"), name="ShutdownOnExitMenu")
                self._running = False  # Signal loop to stop
                return

            # Handle actions requiring auth when not available
            if requires_auth and action_method is None:
                self.ui.print_error("This action requires initialized API credentials and a connection.")
                self.ui.print_warning("Please check your .env file and restart, or check logs for connection errors.")
                await self.ui_wait_for_enter_async()
                return

            # Execute the action if available and method is defined
            if action_method:
                action_name = option_text  # Use menu text as action name for logging/errors
                logger.info(f"Executing action: {action_name}")
                # Call the action handler wrapper
                await self.handle_action(action_method, action_name)
            else:
                # Should not happen if logic is correct (Exit is handled, auth check done)
                self.ui.print_error(f"Selected action '{option_text}' is unexpectedly unavailable or not implemented.")
                await self.ui_wait_for_enter_async()
        else:
            # Should not happen due to menu validation, but good practice
            self.ui.print_error(f"Invalid menu choice received: {choice}")
            await asyncio.sleep(1)

    async def handle_action(self, action_coro: Callable, action_name: str) -> None:
        """Executes a menu action coroutine and handles common exceptions."""
        try:
            await action_coro()
            # Wait is now handled at the end of each action function itself if needed.
            # await self.ui_wait_for_enter_async()

        # --- Specific CCXT Error Handling ---
        except ccxt.AuthenticationError as e:
            self.ui.print_error(f"Authentication Error during '{action_name}': {e}. Check API keys/permissions.")
            await self.ui_wait_for_enter_async()
        except ccxt.InsufficientFunds as e:
            self.ui.print_error(f"Insufficient Funds during '{action_name}': {e}")
            await self.ui_wait_for_enter_async()
        except ccxt.InvalidOrder as e:
            # Log full error for details, show simplified message
            logger.error(f"InvalidOrder during '{action_name}': {e}", exc_info=True)
            self.ui.print_error(
                f"Invalid Order parameters during '{action_name}'. Check details (symbol, size, price, etc.) and exchange rules. See logs for full error. Message: {e}"
            )
            await self.ui_wait_for_enter_async()
        except ccxt.BadRequest as e:  # Catch things like invalid params, account type issues
            logger.error(f"BadRequest during '{action_name}': {e}", exc_info=True)
            self.ui.print_error(
                f"Bad Request during '{action_name}'. Check parameters or account settings. Message: {e}"
            )
            await self.ui_wait_for_enter_async()
        except ccxt.BadSymbol as e:
            self.ui.print_error(
                f"Invalid Symbol during '{action_name}': {e}. Ensure symbol exists and is formatted correctly (e.g., BTC/USDT)."
            )
            await self.ui_wait_for_enter_async()
        except ccxt.NetworkError as e:  # Includes timeouts, DNS issues, etc.
            self.ui.print_error(
                f"Network Error during '{action_name}': {e}. Check internet connection and Bybit status."
            )
            await self.ui_wait_for_enter_async()
        except ccxt.ExchangeNotAvailable as e:
            self.ui.print_error(
                f"Exchange Not Available during '{action_name}': {e}. Bybit might be down for maintenance."
            )
            await self.ui_wait_for_enter_async()
        except ccxt.ExchangeError as e:  # Catch-all for other specific exchange-reported errors
            logger.error(f"ExchangeError during '{action_name}': {e}", exc_info=True)
            self.ui.print_error(f"Exchange Error during '{action_name}': {e}.")
            await self.ui_wait_for_enter_async()

        # --- Application/Input Errors ---
        except ConnectionError as e:  # Raised by _check_initialized or client init failures
            self.ui.print_error(
                f"Connection Error during '{action_name}': {e}. Client might be offline or uninitialized."
            )
            await self.ui_wait_for_enter_async()
        except ValueError as e:  # Catch validation errors etc. from get_input or TA/data processing
            self.ui.print_error(f"Input or Data Error during '{action_name}': {e}")
            # Don't log full trace for simple value errors usually
            logger.error(f"Input/Data ValueError during '{action_name}': {e}")
            await self.ui_wait_for_enter_async()
        except (EOFError, KeyboardInterrupt) as e:
            # These are raised by the async input helpers if encountered during an action
            logger.warning(f"{type(e).__name__} caught during action '{action_name}'. Initiating shutdown.")
            if not self._shutdown_event.is_set():  # Prevent duplicate calls
                self._create_task(self.shutdown(signal=type(e).__name__, exit_code=1), name="ShutdownOnActionInterrupt")
            self._running = False  # Signal loop to stop
            # Don't wait for enter, let shutdown proceed / main loop break
        except Exception as e:
            # Catch any other unexpected errors during the action's execution
            logger.error(f"Unhandled error in action '{action_name}': {e}", exc_info=True)
            self.ui.print_error(f"An unexpected error occurred in '{action_name}': {e}")
            await self.ui_wait_for_enter_async()

    # --- Async Input Helpers (Using run_in_executor) ---

    async def ui_display_menu_async(self) -> str:
        """Displays the main menu and gets choice asynchronously."""
        loop = asyncio.get_running_loop()

        # Determine available actions based on client status
        is_authenticated = self.exchange_client is not None and self.exchange_client._initialized

        # Define menu options text and auth requirement
        all_menu_options_text = [
            ("Place Order", True),
            ("View Account Balance", True),
            ("View Open Orders", True),
            ("View Open Positions", True),
            ("Check Current Price", True),
            ("Technical Analysis", True),  # Requires fetch_ohlcv which needs client
            ("Settings", False),
            ("Exit", False),
        ]

        # Build the list of options to display, marking disabled ones
        current_menu_display_options = []
        for text, requires_auth in all_menu_options_text:
            if requires_auth and not is_authenticated:
                # Append disabled marker using theme colors
                current_menu_display_options.append(
                    f"{text} {self.ui.colors['dim']}(Auth Required){self.ui.colors['menu_option']}"
                )  # Use menu_option color for text part
            else:
                current_menu_display_options.append(text)

        menu_title = f"Pyrmethus Bybit Terminal {'(Testnet)' if self.credentials and self.credentials.testnet else ''}"
        if not is_authenticated:
            # Add warning to title using theme colors
            menu_title += f" {self.ui.colors['warning']}(Limited Mode){self.ui.colors['reset']}"

        # Wrap the blocking UI function call with its arguments using partial
        func_call = partial(self.ui.display_menu, menu_title, current_menu_display_options, "Select your spell")

        # Run the blocking function in the default executor (thread pool)
        try:
            choice = await loop.run_in_executor(None, func_call)
            return choice
        except (EOFError, KeyboardInterrupt) as e:
            logger.warning(f"{type(e).__name__} caught in executor wrapper for display_menu.")
            # Re-raise the specific exception to be caught by the caller (run loop)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in display_menu executor: {e}", exc_info=True)
            # Propagate unexpected errors as RuntimeErrors
            raise RuntimeError(f"Failed to display menu: {e}") from e

    async def ui_get_input_async(self, *args, **kwargs) -> Any:
        """Asynchronously gets validated user input using executor."""
        loop = asyncio.get_running_loop()
        # Wrap the call to self.ui.get_input with its arguments using partial
        func_call = partial(self.ui.get_input, *args, **kwargs)
        try:
            return await loop.run_in_executor(None, func_call)
        except (EOFError, KeyboardInterrupt) as e:
            logger.warning(f"{type(e).__name__} caught in executor wrapper for get_input.")
            # Re-raise to be caught by handle_action or the calling function
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in get_input executor: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get user input: {e}") from e

    async def ui_wait_for_enter_async(self, prompt: str = "Press Enter to continue...") -> None:
        """Asynchronously waits for Enter key using executor."""
        loop = asyncio.get_running_loop()
        # Wrap the call using partial
        func_call = partial(self.ui.wait_for_enter, prompt)
        try:
            await loop.run_in_executor(None, func_call)
        except (EOFError, KeyboardInterrupt) as e:
            logger.warning(f"{type(e).__name__} caught in executor wrapper for wait_for_enter.")
            # Re-raise to be caught by handle_action or the calling function
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in wait_for_enter executor: {e}", exc_info=True)
            raise RuntimeError(f"Failed waiting for Enter: {e}") from e

    # --- Input Validation Helpers ---

    def _validate_symbol(self, symbol: str) -> bool | str:
        """Validates symbol format and checks against loaded markets if available."""
        if not isinstance(symbol, str) or not symbol:
            return "Symbol cannot be empty."
        symbol_upper = symbol.strip().upper()

        # Basic format check (allow letters, numbers, slash, maybe hyphen for some pairs)
        # Adjusted to be slightly more permissive, rely more on market check
        if not re.match(r"^[A-Z0-9/-]{3,25}$", symbol_upper):
            return f"Symbol '{symbol}' has invalid characters or length."

        # Check format if slash is present (CCXT standard)
        if "/" in symbol_upper:
            parts = symbol_upper.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                return "Invalid format. Use BASE/QUOTE (e.g., BTC/USDT)."
        # If no slash, we'll try to normalize it later, basic length check done above

        # Validate against loaded markets if available and client is initialized
        if self.exchange_client and self.exchange_client._markets_loaded:
            ccxt_symbol = self._get_ccxt_symbol(symbol)  # Attempt to normalize format
            if not self.exchange_client.exchange or not self.exchange_client.exchange.markets:
                logger.warning("Markets object missing during symbol validation.")
                return True  # Cannot validate further, assume okay for now

            if ccxt_symbol not in self.exchange_client.exchange.markets:
                # Try to provide helpful suggestions
                available_symbols = list(self.exchange_client.exchange.markets.keys())
                usdt_symbols = sorted([s for s in available_symbols if isinstance(s, str) and s.endswith("/USDT")])
                suggestions = []
                # Try matching start or if user omitted slash
                base_attempt = symbol_upper.replace("/", "")  # Remove slash for comparison
                suggestions = [
                    s for s in usdt_symbols if s.startswith(base_attempt[:3]) or s.replace("/", "") == base_attempt
                ]

                suggestion_str = ""
                if suggestions:
                    suggestion_str = f" Did you mean one of these? {', '.join(suggestions[:5])}..."
                elif usdt_symbols:
                    suggestion_str = f" Available USDT examples: {', '.join(usdt_symbols[:5])}..."

                return f"Symbol '{ccxt_symbol}' not found on loaded exchange markets.{suggestion_str}"

        return True  # Passed basic checks or market check passed/unavailable

    def _get_ccxt_symbol(self, user_input: str) -> str:
        """Converts user input (e.g., BTCUSDT or BTC/USDT) to CCXT standard format (BTC/USDT)."""
        if not isinstance(user_input, str):
            return ""  # Handle non-string input
        symbol = user_input.strip().upper()
        if "/" in symbol:
            return symbol  # Already in correct format

        # If markets loaded, try to find the CCXT symbol matching the input (more reliable)
        if (
            self.exchange_client
            and self.exchange_client._markets_loaded
            and self.exchange_client.exchange
            and self.exchange_client.exchange.markets
        ):
            # Check if input matches an exchange-specific ID ('id' field in market info)
            for market_symbol, market_data in self.exchange_client.exchange.markets.items():
                if market_data.get("id", "").upper() == symbol:
                    # Ensure it's a USDT perpetual swap (linear)
                    if market_data.get("swap") and market_data.get("linear") and market_data.get("quote") == "USDT":
                        logger.debug(
                            f"Formatted symbol '{symbol}' to '{market_symbol}' based on market ID (USDT Swap)."
                        )
                        return market_symbol
                    else:
                        logger.debug(
                            f"Symbol '{symbol}' matched market ID '{market_symbol}' but it's not a USDT Perpetual Swap. Skipping."
                        )

            # Check if input matches concatenated BASEQUOTE for USDT swaps
            possible_matches = []
            for market_symbol, market_info in self.exchange_client.exchange.markets.items():
                # Ensure 'base' and 'quote' are available and it's a USDT swap
                base = market_info.get("base")
                quote = market_info.get("quote")
                is_swap = market_info.get("swap", False)
                is_linear = market_info.get("linear", False)

                if base and quote and is_swap and is_linear and quote == "USDT":
                    if symbol == f"{base}{quote}":
                        possible_matches.append(market_symbol)

            if possible_matches:
                formatted = possible_matches[0]  # Usually only one match for BASEQUOTE
                logger.debug(f"Formatted symbol '{symbol}' to '{formatted}' based on concatenated market (USDT Swap).")
                return formatted

        # Generic guess if markets not loaded or no match found (less reliable)
        if symbol.endswith("USDT") and len(symbol) > 4:
            base = symbol[:-4]
            formatted = f"{base}/USDT"
            logger.debug(f"Formatted symbol '{symbol}' to '{formatted}' (generic USDT guess).")
            return formatted

        logger.warning(
            f"{Fore.YELLOW}Could not reliably format symbol '{symbol}' to BASE/USDT swap format. Using as is. Exchange might reject it.{Style.RESET_ALL}"
        )
        return symbol  # Return original if no formatting rule applied

    def _validate_side(self, side: str) -> bool | str:
        """Validates order side."""
        if not isinstance(side, str) or side.lower() not in ["buy", "sell"]:
            return "Invalid side. Must be 'buy' or 'sell'."
        return True

    def _validate_order_type(self, order_type: str) -> bool | str:
        """Validates order type."""
        # Expand supported types if exchange/CCXT allows more easily (e.g., StopLimit)
        supported_types = ["market", "limit"]  # Keep it simple for now
        if not isinstance(order_type, str) or order_type.lower() not in supported_types:
            supported_str = ", ".join(t.capitalize() for t in supported_types)
            return f"Invalid order type. Supported: {supported_str}."
        return True

    def _validate_positive_float(self, value: Any) -> bool | str:
        """Validates if the value is a float strictly greater than zero."""
        if value is None:
            return "Input cannot be empty/None."  # Handle None explicitly
        try:
            num_value = float(value)
            if not np.isfinite(num_value):
                return "Value must be a finite number."
            if num_value <= 0:
                return "Value must be a positive number (greater than 0)."
            return True
        except (ValueError, TypeError):
            return "Input must be a valid number (e.g., 0.5, 100)."

    def _validate_non_negative_float(self, value: Any) -> bool | str:
        """Validates if the value is a float greater than or equal to zero."""
        if value is None:
            return "Input cannot be empty/None."  # Handle None explicitly
        try:
            num_value = float(value)
            if not np.isfinite(num_value):
                return "Value must be a finite number."
            if num_value < 0:
                return "Value cannot be negative."
            return True
        except (ValueError, TypeError):
            return "Input must be a valid number (e.g., 0, 0.5, 100)."

    def _validate_percentage(self, value: Any) -> bool | str:
        """Validate percentage, usually > 0 and <= 100 for trailing stops."""
        result = self._validate_positive_float(value)  # Must be positive
        if result is not True:
            return result  # Inherit error message from positive float check
        num_value = float(value)
        # Set a reasonable upper limit (e.g., 100%, maybe less depending on context)
        if num_value > 100:
            return "Percentage cannot exceed 100."
        # Set a practical lower limit (e.g., 0.01% for Bybit)
        if num_value < 0.01:
            return "Percentage seems too small (Bybit min is typically 0.01%)."
        return True

    def _validate_timeframe(self, timeframe: str) -> bool | str:
        """Basic validation for common CCXT timeframe formats and checks against exchange."""
        if not timeframe:
            return "Timeframe cannot be empty."
        timeframe_in = str(timeframe).lower()  # Normalize to lowercase for checks

        # Regex for common CCXT styles (e.g., 1m, 5m, 1h, 4h, 1d, 1w, 1M)
        # Allow 1-3 digits for number part, single char for unit
        if not re.match(r"^\d{1,3}[mhdwyM]$", timeframe_in):
            # Check for Bybit specific non-standard timeframes (D, W, M)
            if timeframe_in.upper() in ["D", "W", "M"]:
                # Consider valid, will be mapped later if needed
                pass
            else:
                return "Invalid timeframe format. Use CCXT style (e.g., '1m', '15m', '1h', '4h', '1d', '1W', '1M') or Bybit single letters (D, W, M)."

        # Map Bybit single letters to CCXT standard for internal consistency check
        tf_map = {"D": "1d", "W": "1w", "M": "1M"}
        timeframe_check = tf_map.get(timeframe_in.upper(), timeframe_in)

        # Optional: Check against exchange.timeframes if markets/client are loaded
        if (
            self.exchange_client
            and self.exchange_client._markets_loaded
            and self.exchange_client.exchange
            and self.exchange_client.exchange.timeframes
        ):
            # Check if the (potentially normalized) timeframe exists
            if timeframe_check not in self.exchange_client.exchange.timeframes:
                available_tfs = sorted(self.exchange_client.exchange.timeframes.keys())
                # Provide closer suggestions if possible
                suggestions = [
                    tf for tf in available_tfs if tf.startswith(timeframe_check[0]) or tf.endswith(timeframe_check[-1])
                ]
                suggestion_str = f" Available examples: {', '.join(available_tfs[:10])}..."
                if suggestions:
                    suggestion_str = f" Did you mean one of these? {', '.join(suggestions[:5])}... Or see examples: {', '.join(available_tfs[:5])}..."

                return f"Timeframe '{timeframe_check}' may not be supported by the exchange.{suggestion_str}"
        return True  # Passed validation

    # --- Menu Actions ---

    async def place_order_menu(self) -> None:
        """Handles the logic for placing an order via user input."""
        self.ui.display_header("Place Order Spell")

        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        # Use async input helper with validation
        symbol_input = await self.ui_get_input_async(
            "Enter symbol sigil (e.g., BTC/USDT or BTCUSDT)",
            default=default_symbol,
            validation_func=self._validate_symbol,
        )
        if symbol_input is None:
            return  # Handle potential None return on optional input or error propagation
        symbol = self._get_ccxt_symbol(symbol_input)  # Normalize after validation passes

        side = await self.ui_get_input_async("Enter side (buy/sell)", validation_func=self._validate_side)
        if side is None:
            return
        side = side.lower()

        default_order_type = self.config_manager.config.get("default_order_type", "Limit")
        order_type = await self.ui_get_input_async(
            "Enter order type (Market/Limit)", default=default_order_type, validation_func=self._validate_order_type
        )
        if order_type is None:
            return
        order_type = order_type.lower()

        # Ensure amount input uses float and positive validator
        amount = await self.ui_get_input_async(
            "Enter quantity (in base currency, e.g., BTC)",
            input_type=float,
            validation_func=self._validate_positive_float,
        )
        if amount is None:
            return

        price = None
        params = {}  # Dictionary for CCXT standard order parameters + exchange specifics

        if order_type == "limit":
            price = await self.ui_get_input_async(
                "Enter limit price", input_type=float, validation_func=self._validate_positive_float
            )
            if price is None:
                return  # Price required for limit order

        # --- Advanced Order Parameters ---
        self.ui.print_info("\nConfigure optional parameters (leave blank or 0 for none):")

        # Stop Loss (Trigger Price) - Use non-negative float validator
        sl_price = await self.ui_get_input_async(
            "Stop Loss trigger price",
            default="0",
            required=False,
            input_type=float,
            validation_func=self._validate_non_negative_float,
        )
        if sl_price is not None and sl_price > 0:
            params["stopLossPrice"] = sl_price  # CCXT standard key for trigger price
            # Optional: Add trigger type input if needed
            # sl_trigger = await self.ui_get_input_async("SL Trigger By (MarkPrice/LastPrice/IndexPrice)", default="MarkPrice")
            # params['slTriggerBy'] = sl_trigger # Example Bybit V5 param
            self.ui.print_info(f"Stop Loss trigger set at {sl_price}")

        # Take Profit (Trigger Price) - Use non-negative float validator
        tp_price = await self.ui_get_input_async(
            "Take Profit trigger price",
            default="0",
            required=False,
            input_type=float,
            validation_func=self._validate_non_negative_float,
        )
        if tp_price is not None and tp_price > 0:
            params["takeProfitPrice"] = tp_price  # CCXT standard key for trigger price
            # Optional: Add trigger type input if needed
            # tp_trigger = await self.ui_get_input_async("TP Trigger By (MarkPrice/LastPrice/IndexPrice)", default="MarkPrice")
            # params['tpTriggerBy'] = tp_trigger # Example Bybit V5 param
            self.ui.print_info(f"Take Profit trigger set at {tp_price}")

        # Trailing Stop (Percentage) - Use percentage validator
        trail_percent = await self.ui_get_input_async(
            "Trailing Stop percentage (e.g., 0.5 for 0.5%, min 0.01)",
            default="0",
            required=False,
            input_type=float,
            validation_func=self._validate_percentage,
        )
        if trail_percent is not None and trail_percent > 0:
            # Use standard CCXT param 'trailingPercent' (positive value)
            params["trailingPercent"] = trail_percent
            # Note: Bybit V5 might need activation price or other params if CCXT mapping fails.
            # Consider adding input for activation price if issues arise.
            self.ui.print_info(f"Trailing Stop set at {trail_percent}%. Activation details depend on exchange/CCXT.")

        # Reduce Only
        reduce_only_str = await self.ui_get_input_async(
            "Set as Reduce Only? (yes/no)",
            default="no",
            required=False,
            validation_func=lambda s: s.lower() in ["yes", "no"] or "Must be 'yes' or 'no'",
        )
        if reduce_only_str and reduce_only_str.lower() == "yes":
            params["reduceOnly"] = True
            self.ui.print_info("Order set to Reduce Only.")

        # --- Confirmation ---
        confirm_parts = [f"Confirm: Place {order_type.upper()} {side.upper()} order for {amount} {symbol}"]
        if price:
            confirm_parts.append(f"at price {price}")
        if "stopLossPrice" in params:
            confirm_parts.append(f"with SL @ {params['stopLossPrice']}")
        if "takeProfitPrice" in params:
            confirm_parts.append(f"with TP @ {params['takeProfitPrice']}")
        if "trailingPercent" in params:
            confirm_parts.append(f"with Trail {params['trailingPercent']}%")
        if params.get("reduceOnly"):
            confirm_parts.append("(Reduce Only)")
        " ".join(confirm_parts) + "?"

        confirm = await self.ui_get_input_async("Type 'yes' to confirm", default="no")

        if confirm is None or confirm.lower() != "yes":
            self.ui.print_warning("Order spell cancelled.")
            await self.ui_wait_for_enter_async()
            return

        # --- Place the Order ---
        self.ui.print_info(f"{self.ui.colors['primary']}# Submitting order incantation...{self.ui.colors['reset']}")
        # Call the client method directly, errors will be caught by handle_action
        result = await self.exchange_client.place_order(symbol, side, order_type, amount, price, params)

        self.ui.print_success("Order submission attempt processed by exchange.")

        # Display selected details from the result dict using print_table for dict
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
        ]
        # Add keys for params passed, check if they exist in the standard response
        # Note: CCXT might not return trigger prices in the immediate response, they might be part of the order status later.
        if "stopLossPrice" in params:
            display_keys.append("stopLossPrice")
        if "takeProfitPrice" in params:
            display_keys.append("takeProfitPrice")
        if "trailingPercent" in params:
            display_keys.append("trailingPercent")
        if "reduceOnly" in params:
            display_keys.append("reduceOnly")

        display_result = {}
        for k in display_keys:
            value = result.get(k)
            if value is not None:
                # Format fee nicely
                if k == "fee" and isinstance(value, dict):
                    fee_info = value
                    cost = fee_info.get("cost")
                    currency = fee_info.get("currency")
                    display_result[k] = f"{cost:.8f} {currency}" if cost is not None and currency else fee_info
                # Format datetime
                elif k == "datetime" and value:
                    try:
                        display_result[k] = pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S UTC")
                    except:
                        display_result[k] = value  # Fallback if parsing fails
                # Format floats with appropriate precision
                elif isinstance(value, (float, np.floating)):
                    precision = 8 if k in ["price", "amount", "cost", "filled", "remaining", "average"] else 4
                    display_result[k] = f"{value:.{precision}f}"
                else:
                    display_result[k] = value

        # Check 'info' for potentially missing details like trigger prices or trailing stop status if not in standard keys
        if "info" in result and isinstance(result["info"], dict):
            info_data = result["info"]
            # Add Bybit specific trigger/trail info if not found in standard keys
            if "stopLossPrice" not in display_result and info_data.get("stopLoss"):
                display_result["SL Price (Info)"] = info_data["stopLoss"]
            if "takeProfitPrice" not in display_result and info_data.get("takeProfit"):
                display_result["TP Price (Info)"] = info_data["takeProfit"]
            if "trailingPercent" not in display_result and info_data.get("trailingStop"):
                display_result["Trail Info"] = info_data["trailingStop"]
            if "reduceOnly" not in display_result and info_data.get("reduceOnly") is not None:
                display_result["Reduce Only (Info)"] = info_data["reduceOnly"]
            if "status" not in display_result and info_data.get("orderStatus"):
                display_result["Status (Info)"] = info_data["orderStatus"]

        self.ui.print_table(
            display_result, title="Order Submission Result Details", float_format="{:.8f}"
        )  # Higher precision default
        await self.ui_wait_for_enter_async()

    async def view_balance(self) -> None:
        """Displays account balances using CCXT (handles V5 structure)."""
        self.ui.display_header("Account Balance Scrying (USDT Futures)")
        self.ui.print_info(f"{self.ui.colors['primary']}# Fetching balance essence...{self.ui.colors['reset']}")

        # Fetch balance (client method handles V5 params and parsing)
        balance_data = await self.exchange_client.fetch_balance()

        if not balance_data or not isinstance(balance_data, dict):
            self.ui.print_warning("No detailed USDT balance data found or format unexpected.")
            if balance_data:  # Print raw if something was received but not usable
                self.ui.print_info("Raw balance data received:")
            await self.ui_wait_for_enter_async()
            return

        # Use standard CCXT keys ('free', 'used', 'total') populated by fetch_balance
        display_data = {
            "Available (Free)": balance_data.get("free"),
            "Used Margin": balance_data.get("used"),
            "Total Equity": balance_data.get("total"),
        }

        # Extract PNL from standardized keys populated by fetch_balance
        unrealized_pnl = balance_data.get("unrealizedPnl")
        realized_pnl = balance_data.get("cumRealisedPnl")  # Check if fetch_balance provides this

        # Format PNL with color
        try:
            if unrealized_pnl is not None:
                pnl_float = float(unrealized_pnl)
                pnl_color = (
                    self.ui.colors["positive"]
                    if pnl_float > 0
                    else self.ui.colors["negative"]
                    if pnl_float < 0
                    else self.ui.colors["neutral"]
                )
                display_data["Unrealized PNL (Account)"] = f"{pnl_color}{pnl_float:.4f}{self.ui.colors['reset']}"
            else:
                display_data["Unrealized PNL (Account)"] = "N/A"  # Explicitly show N/A

            if realized_pnl is not None:
                # Use neutral color for realized PNL? Or format based on value?
                rpnl_float = float(realized_pnl)
                self.ui.colors["positive"] if rpnl_float > 0 else self.ui.colors[
                    "negative"
                ] if rpnl_float < 0 else self.ui.colors["neutral"]
                # display_data["Realized PNL (Account)"] = f"{rpnl_color}{rpnl_float:.4f}{self.ui.colors['reset']}" # Optional color
                display_data["Realized PNL (Account)"] = f"{rpnl_float:.4f}"  # Default: no color
            else:
                display_data["Realized PNL (Account)"] = "N/A"

        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse PNL from balance info: {e}")
            if "Unrealized PNL (Account)" not in display_data:
                display_data["Unrealized PNL (Account)"] = f"Error: {unrealized_pnl}"
            if "Realized PNL (Account)" not in display_data:
                display_data["Realized PNL (Account)"] = f"Error: {realized_pnl}"

        # Filter out None values before printing for cleaner display
        # Keep N/A strings we added
        display_data_filtered = {k: v for k, v in display_data.items() if v is not None}

        if not display_data_filtered or not any(
            k in display_data_filtered for k in ["Available (Free)", "Used Margin", "Total Equity"]
        ):
            self.ui.print_warning("Balance data received but key fields (free, used, total) are missing or null.")
            self.ui.print_info("Raw balance data:")
        else:
            # Let print_table handle the dictionary with colored strings directly
            self.ui.print_table(display_data_filtered, title="USDT Futures Account Balance", float_format="{:.4f}")

            self.ui.print_info("\nNote: PNL figures are account-wide estimates from balance data, if available.")
            self.ui.print_info("Use 'View Open Positions' for PNL per position.")

        await self.ui_wait_for_enter_async()

    async def view_open_orders(self) -> None:
        """Fetches and displays open orders."""
        self.ui.display_header("View Open Order Scrolls")
        # Allow empty input for 'all symbols'
        symbol_input = await self.ui_get_input_async(
            "Enter symbol to filter (e.g., BTC/USDT) or leave blank for all",
            required=False,
            validation_func=lambda s: True if not s else self._validate_symbol(s),
        )
        symbol = self._get_ccxt_symbol(symbol_input) if symbol_input else None

        action = f"for {symbol}" if symbol else "for all symbols"
        self.ui.print_info(
            f"{self.ui.colors['primary']}# Fetching active order scrolls {action}...{self.ui.colors['reset']}"
        )

        open_orders = await self.exchange_client.fetch_open_orders(symbol=symbol)

        if not open_orders:
            self.ui.print_warning(f"No open orders found {action}.")
        else:
            # Select, rename, and format columns for clarity using standard CCXT keys
            display_data = []
            for order in open_orders:
                order_info = {
                    "ID": order.get("id"),
                    "Timestamp": pd.to_datetime(order.get("datetime")).strftime("%Y-%m-%d %H:%M:%S")
                    if order.get("datetime")
                    else "N/A",
                    "Symbol": order.get("symbol"),
                    "Type": order.get("type", "").capitalize(),
                    "Side": order.get("side", "").capitalize(),
                    "Price": order.get("price"),
                    "Amount": order.get("amount"),
                    "Filled": order.get("filled"),
                    "Remaining": order.get("remaining"),
                    "Status": order.get("status"),
                    # Standard CCXT keys for triggers (might be None if not set or not returned)
                    "SL Price": order.get("stopLossPrice"),
                    "TP Price": order.get("takeProfitPrice"),
                    "Trail %": order.get("trailingPercent"),  # Standard key, might be None
                    "ReduceOnly": order.get("reduceOnly"),
                }

                # Add specific Trail info from 'info' if standard key is missing/None
                trail_info_str = "N/A"
                if order_info["Trail %"] is None and "info" in order and isinstance(order["info"], dict):
                    trail_info_raw = order["info"].get("trailingStop")  # Bybit V5 key
                    if trail_info_raw:
                        # Bybit often returns string "0" if not set, handle that
                        if trail_info_raw != "0":
                            trail_info_str = str(trail_info_raw)  # Show the raw value
                            # Try to parse if it looks like a dict (less common now?)
                            # if isinstance(trail_info_raw, str) and trail_info_raw.startswith('{'): ...
                elif order_info["Trail %"] is not None:
                    trail_info_str = f"{order_info['Trail %']}%"  # Format standard percentage

                order_info["Trail Info"] = trail_info_str  # Add consolidated trail info

                # Remove the original Trail % if we added Trail Info
                if "Trail %" in order_info:
                    del order_info["Trail %"]

                # Add Trigger info from 'info' if standard keys are missing
                if order_info["SL Price"] is None and order["info"].get("stopLoss"):
                    order_info["SL Price"] = order["info"]["stopLoss"]
                if order_info["TP Price"] is None and order["info"].get("takeProfit"):
                    order_info["TP Price"] = order["info"]["takeProfit"]

                display_data.append(order_info)

            # Use higher precision for price/amount in crypto
            self.ui.print_table(display_data, title=f"Open Orders {action}", float_format="{:.8f}")

        await self.ui_wait_for_enter_async()

    async def view_open_positions(self) -> None:
        """Fetches and displays open positions."""
        self.ui.display_header("View Open Position Essences")
        symbol_input = await self.ui_get_input_async(
            "Enter symbol to filter (e.g., BTC/USDT) or leave blank for all",
            required=False,
            validation_func=lambda s: True if not s else self._validate_symbol(s),
        )
        symbol = self._get_ccxt_symbol(symbol_input) if symbol_input else None

        action = f"for {symbol}" if symbol else "for all symbols"
        self.ui.print_info(
            f"{self.ui.colors['primary']}# Fetching open position essences {action}...{self.ui.colors['reset']}"
        )

        # Fetch positions (client method already filters for non-zero size)
        positions = await self.exchange_client.fetch_positions(symbol=symbol)

        if not positions:
            self.ui.print_warning(f"No open positions found {action}.")
        else:
            # Select, rename, and format columns using CCXT standard keys
            display_data = []
            for pos in positions:
                # Extract data using standard CCXT keys
                symbol = pos.get("symbol")
                side = pos.get("side")  # 'long' or 'short'
                # Use the size parsing logic from fetch_positions again for consistency
                size_str = None
                if pos.get("contracts") is not None:
                    size_str = str(pos["contracts"])
                elif pos.get("contractSize") is not None:
                    size_str = str(pos["contractSize"])
                elif "info" in pos and pos["info"] and pos["info"].get("size") is not None:
                    size_str = str(pos["info"]["size"])
                size = float(size_str) if size_str is not None and size_str != "" else 0.0  # Convert to float

                entry_price = pos.get("entryPrice")
                mark_price = pos.get("markPrice")
                liq_price = pos.get("liquidationPrice")
                initial_margin = pos.get("initialMargin")
                pos.get("maintenanceMargin")
                unrealized_pnl = pos.get("unrealizedPnl")
                leverage = pos.get("leverage")
                notional = pos.get("notional")  # Size * Price

                pos_info = {
                    "Symbol": symbol,
                    "Side": side.capitalize() if side else "N/A",
                    "Size": size,
                    "Entry Price": entry_price,
                    "Mark Price": mark_price,
                    "Liq Price": liq_price,
                    "Margin": initial_margin,
                    "uPNL": unrealized_pnl,  # Will be formatted with color below
                    "uPNL %": "N/A",  # Placeholder, calculated below
                    "Leverage": f"{leverage}x" if leverage else "N/A",
                    "Notional": notional,
                    # "Maint Margin": maint_margin, # Can be verbose, optional
                }

                # Calculate percentage PNL based on initial margin if possible
                pnl_percent_str = "N/A"
                if initial_margin is not None and initial_margin != 0 and unrealized_pnl is not None:
                    try:
                        pnl_percent = (float(unrealized_pnl) / float(initial_margin)) * 100
                        pnl_color = (
                            self.ui.colors["positive"]
                            if pnl_percent > 0
                            else self.ui.colors["negative"]
                            if pnl_percent < 0
                            else self.ui.colors["neutral"]
                        )
                        # Store the colored string for display
                        pnl_percent_str = f"{pnl_color}{pnl_percent:.2f}%{self.ui.colors['reset']}"
                    except (ValueError, TypeError, ZeroDivisionError):
                        pnl_percent_str = "Error"  # Indicate calculation error

                pos_info["uPNL %"] = pnl_percent_str

                # Color uPNL value itself
                if unrealized_pnl is not None:
                    try:
                        pnl_float = float(unrealized_pnl)
                        pnl_color = (
                            self.ui.colors["positive"]
                            if pnl_float > 0
                            else self.ui.colors["negative"]
                            if pnl_float < 0
                            else self.ui.colors["neutral"]
                        )
                        # Override the uPNL value with the colored string
                        pos_info["uPNL"] = f"{pnl_color}{pnl_float:.4f}{self.ui.colors['reset']}"
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

                # Add the processed dict to the list for the table
                display_data.append(pos_info)

            # Print table using the enhanced print_table which handles colored strings
            self.ui.print_table(display_data, title=f"Open Positions {action}", float_format="{:.4f}")

        await self.ui_wait_for_enter_async()

    async def check_price_menu(self) -> None:
        """Fetches and displays the current ticker price."""
        self.ui.display_header("Check Price Pulse")
        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        symbol_input = await self.ui_get_input_async(
            "Enter symbol sigil (e.g., BTC/USDT or BTCUSDT)",
            default=default_symbol,
            validation_func=self._validate_symbol,
        )
        if symbol_input is None:
            return
        symbol = self._get_ccxt_symbol(symbol_input)

        self.ui.print_info(
            f"{self.ui.colors['primary']}# Fetching current pulse for {symbol}...{self.ui.colors['reset']}"
        )
        ticker = await self.exchange_client.fetch_ticker(symbol)

        # Display relevant ticker info using print_table for dict
        display_data = {
            "Symbol": ticker.get("symbol"),
            "Timestamp": pd.to_datetime(ticker.get("datetime")).strftime("%Y-%m-%d %H:%M:%S UTC")
            if ticker.get("datetime")
            else "N/A",
            "Last Price": ticker.get("last"),
            "Mark Price": ticker.get("mark"),
            "Index Price": ticker.get("index"),
            "Bid Price": ticker.get("bid"),
            "Ask Price": ticker.get("ask"),
            "High (24h)": ticker.get("high"),
            "Low (24h)": ticker.get("low"),
            "Volume (24h Base)": ticker.get("baseVolume"),  # e.g., BTC
            "Volume (24h Quote)": ticker.get("quoteVolume"),  # e.g., USDT
            "Change (24h)": ticker.get("change"),
            "% Change (24h)": ticker.get("percentage"),  # Will be formatted with color
            "Funding Rate (Est)": ticker.get("fundingRate"),  # Standard CCXT key
            "Next Funding Time": pd.to_datetime(ticker.get("fundingTimestamp"), unit="ms").strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            if ticker.get("fundingTimestamp")
            else None,
        }

        # Colorize % Change
        perc_change = display_data.get("% Change (24h)")
        if perc_change is not None:
            try:
                perc_float = float(perc_change)
                perc_color = (
                    self.ui.colors["positive"]
                    if perc_float > 0
                    else self.ui.colors["negative"]
                    if perc_float < 0
                    else self.ui.colors["neutral"]
                )
                display_data["% Change (24h)"] = f"{perc_color}{perc_float:.2f}%{self.ui.colors['reset']}"
            except (ValueError, TypeError):
                pass  # Keep original if not a number

        # Format Funding Rate if available
        funding_rate = display_data.get("Funding Rate (Est)")
        if funding_rate is not None:
            try:
                display_data["Funding Rate (Est)"] = f"{float(funding_rate) * 100:.4f}%"  # Convert to percentage
            except (ValueError, TypeError):
                pass  # Keep original if not a number

        # Filter None values before printing for cleaner display
        display_data_filtered = {k: v for k, v in display_data.items() if v is not None}

        # Let print_table handle the dictionary with colored strings
        self.ui.print_table(display_data_filtered, title=f"Current Ticker: {symbol}", float_format="{:.4f}")
        await self.ui_wait_for_enter_async()

    async def technical_analysis_menu(self) -> None:
        """Handles fetching data and displaying technical analysis."""
        if ta is None:
            self.ui.print_error("Technical Analysis requires the 'pandas_ta' library.")
            self.ui.print_warning("Please install it using: pip install pandas_ta")
            await self.ui_wait_for_enter_async()
            return

        self.ui.display_header("Technical Analysis Oracle")

        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        symbol_input = await self.ui_get_input_async(
            "Enter symbol sigil (e.g., BTC/USDT or BTCUSDT)",
            default=default_symbol,
            validation_func=self._validate_symbol,
        )
        if symbol_input is None:
            return
        symbol = self._get_ccxt_symbol(symbol_input)

        default_timeframe = self.config_manager.config.get("default_timeframe", "1h")
        timeframe_input = await self.ui_get_input_async(
            "Enter time-shard (e.g., 1m, 5m, 1h, 1d)",
            default=default_timeframe,
            validation_func=self._validate_timeframe,
        )
        if timeframe_input is None:
            return
        # Normalize timeframe after validation passes
        timeframe = timeframe_input.lower()
        tf_map = {"d": "1d", "w": "1w", "m": "1M"}  # Map Bybit letters if used
        if timeframe.upper() in tf_map:
            timeframe = tf_map[timeframe.upper()]

        # Determine data fetch limit
        chart_points = self.config_manager.config.get("chart_points", 60)
        # Estimate max lookback needed for indicators + buffer for warmup/NaNs
        indicator_periods = self.config_manager.config.get("indicator_periods", {})
        max_lookback_needed = 0
        try:
            # Use configured periods, provide large defaults if missing
            max_lookback_needed = max(
                int(indicator_periods.get("sma_long", 50)),  # SMA needs length
                int(indicator_periods.get("ema", 20)),  # EMA needs length
                int(indicator_periods.get("rsi", 14)),  # RSI needs length + buffer
                int(indicator_periods.get("bbands", 20)),  # BBands needs length
                26 + 9,  # MACD default fast + signal periods
                int(indicator_periods.get("stoch_k", 14))
                + int(indicator_periods.get("stoch_d", 3))
                + int(indicator_periods.get("stoch_smooth_k", 3)),  # Stoch needs k+d+smooth_k
                int(indicator_periods.get("atr", 14)),  # ATR needs length
            )
            # Add a buffer (e.g., 50 candles) for NaN warmup
            max_lookback_needed += 50
        except Exception as e:
            logger.warning(f"Could not calculate max indicator lookback: {e}. Using default 250.")
            max_lookback_needed = 250  # Default large lookback

        # Fetch enough data: max(chart points, indicator lookback) + buffer
        # CCXT limit parameter might have exchange restrictions (e.g., Bybit max 1000 or 200 for older API)
        # Fetch slightly more than needed, respecting potential exchange limits
        fetch_limit = min(max(chart_points + 50, max_lookback_needed + 50), 1000)  # Add buffer, cap at 1000

        self.ui.print_info(
            f"{self.ui.colors['primary']}# Scrying market data for {symbol} ({timeframe}, limit={fetch_limit})...{self.ui.colors['reset']}"
        )
        ohlcv_data = await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)

        if not ohlcv_data:
            self.ui.print_warning(f"No OHLCV data received for {symbol} ({timeframe}). Cannot perform analysis.")
            await self.ui_wait_for_enter_async()
            return

        # --- Process Data ---
        df = None  # Initialize df to None
        try:
            df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # Convert timestamp to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            # Convert OHLCV columns to numeric, coercing errors (handles None, strings)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Log how many rows were received *before* cleaning
            logger.info(f"Received {len(df)} data points from exchange.")

            # Drop rows where essential data (close) is missing AFTER conversion
            initial_len = len(df)
            df.dropna(subset=["close", "high", "low"], inplace=True)  # Need HLC for most indicators
            if len(df) < initial_len:
                logger.warning(f"Dropped {initial_len - len(df)} rows with NaN OHLC data during processing.")
            if df.empty:
                raise ValueError("DataFrame is empty after removing NaN OHLC data.")
            logger.info(f"Processed {len(df)} data points after cleaning.")
        except Exception as e:
            logger.error(f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
            self.ui.print_error(f"Failed to process market data: {e}")
            await self.ui_wait_for_enter_async()
            return

        # --- Calculate Indicators ---
        # Pass the cleaned df to the calculator
        df_indicators = self.ta_analyzer.calculate_indicators(df)
        # Check if indicators were actually calculated (TA func logs details)
        if df_indicators is df:  # Check if the original df was returned due to error/no-op
            self.ui.print_error("Technical analysis calculation failed or produced no indicators. Check logs.")
            await self.ui_wait_for_enter_async()
            return
        if df_indicators.empty:  # Check if TA returned an empty dataframe
            self.ui.print_error("Technical analysis calculation resulted in an empty DataFrame.")
            await self.ui_wait_for_enter_async()
            return

        # --- Calculate Pivot Points ---
        pivot_points = None
        pivot_period_str = self.config_manager.config.get(
            "pivot_period", "1d"
        )  # Already normalized in config load/save
        try:
            self.ui.print_info(
                f"{self.ui.colors['primary']}# Fetching data for {pivot_period_str} pivot points...{self.ui.colors['reset']}"
            )
            # Ensure the pivot timeframe is valid before fetching
            validation_result = self._validate_timeframe(pivot_period_str)
            if validation_result is not True:
                raise ValueError(f"Invalid pivot timeframe in config: {pivot_period_str}. {validation_result}")

            # Fetch last 2 candles. Index 0 is the last *completed* candle. Index 1 is the current (incomplete).
            # Need to handle potential rate limits if called right after main data fetch
            await asyncio.sleep(0.5)  # Small delay
            pivot_ohlcv = await self.exchange_client.fetch_ohlcv(symbol, pivot_period_str, limit=2)

            if pivot_ohlcv and len(pivot_ohlcv) >= 1:
                # Create DataFrame for the pivot period data
                df_pivot_period = pd.DataFrame(
                    pivot_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df_pivot_period["timestamp"] = pd.to_datetime(df_pivot_period["timestamp"], unit="ms")
                df_pivot_period.set_index("timestamp", inplace=True)
                # Convert to numeric, coerce errors
                for col in ["open", "high", "low", "close", "volume"]:
                    df_pivot_period[col] = pd.to_numeric(df_pivot_period[col], errors="coerce")

                # Use the first row fetched (index 0), which is the last completed period
                if not df_pivot_period.empty and len(df_pivot_period) >= 1:
                    # Pass DataFrame slice containing only the row for calculation
                    pivot_calc_data = df_pivot_period.iloc[[0]]  # Pass as DataFrame slice
                    pivot_points = self.ta_analyzer.calculate_pivot_points(pivot_calc_data)
                    if pivot_points:
                        candle_time_str = (
                            df_pivot_period.index[0].strftime("%Y-%m-%d %H:%M:%S")
                            if isinstance(df_pivot_period.index[0], pd.Timestamp)
                            else str(df_pivot_period.index[0])
                        )
                        logger.info(f"Calculated {pivot_period_str} pivots based on candle ending {candle_time_str}")
                    else:
                        # Calc function already logged NaN issue
                        logger.warning(f"Pivot calculation returned None for {pivot_period_str}. Check logs.")
                else:
                    logger.warning(
                        f"Could not use fetched data for {pivot_period_str} pivots after cleaning (NaNs or insufficient rows)."
                    )
            else:
                logger.warning(
                    f"Could not fetch sufficient data (need >= 1 candle) for {pivot_period_str} pivot points."
                )

        except ValueError as e:  # Catch specific errors like invalid timeframe for pivots
            logger.error(f"Error fetching/calculating pivot points ({pivot_period_str}): {e}")
            self.ui.print_warning(f"Could not calculate pivot points ({pivot_period_str}): {e}")
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(f"CCXT Error fetching pivot data ({pivot_period_str}): {e}")
            self.ui.print_warning(f"Could not fetch data for pivot points ({pivot_period_str}): {e}")
        except Exception as e:  # Catch unexpected errors
            logger.error(f"Unexpected error fetching/calculating pivot points: {e}", exc_info=True)
            self.ui.print_warning(f"Could not calculate pivot points ({pivot_period_str}): Unexpected error.")

        # --- Display Results ---

        # TA Summary (from the last row of df_indicators)
        # Use the generate_ta_summary which finds the last valid row
        ta_summary = self.ta_analyzer.generate_ta_summary(df_indicators)
        self.ui.print_table(
            ta_summary, title=f"TA Summary: {symbol} ({timeframe})"
        )  # Let print_table handle dict/colors

        # Pivot Points
        if pivot_points:
            self.ui.print_table(
                pivot_points,
                title=f"Classic Pivot Points (Based on last completed {pivot_period_str})",
                float_format="{:.4f}",
            )
        else:
            self.ui.print_warning(f"Pivot points ({pivot_period_str}) could not be calculated or data unavailable.")

        # Chart (using close prices from df_indicators)
        # Ensure enough valid data points exist before slicing/charting
        close_prices = df_indicators["close"].dropna().tolist()  # Drop NaNs before charting
        if close_prices:
            points_to_chart = min(len(close_prices), chart_points)
            if points_to_chart < chart_points and len(close_prices) > 0:
                self.ui.print_warning(
                    f"Insufficient data ({len(close_prices)} points) for requested chart length ({chart_points}). Charting last {points_to_chart}."
                )

            self.ui.display_chart(
                close_prices[-points_to_chart:],
                f"{symbol} ({timeframe}) Close Price Trend (Last {points_to_chart} points)",
            )
        else:
            self.ui.print_warning("No valid close price data available for chart.")

        # Detailed Indicator Table (Last N periods)
        # Address the "DataFrame boolean ambiguity" error source
        num_detail_rows = 10
        if not df_indicators.empty and len(df_indicators) >= num_detail_rows:
            # Take tail, copy to avoid warnings, format index, make index a column
            display_df = df_indicators.tail(num_detail_rows).copy()
            if isinstance(display_df.index, pd.DatetimeIndex):
                display_df.index = display_df.index.strftime("%m-%d %H:%M")  # Shorter format for table
            display_df = display_df.reset_index()  # Make timestamp a column

            # Select and reorder columns (use lowercase names from processing/TA)
            base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            # Get indicator columns (those not in base_cols) dynamically
            indicator_cols = sorted([col for col in display_df.columns if col not in base_cols])
            # Filter out columns that might not exist if TA failed partially or were removed
            final_cols = [col for col in base_cols + indicator_cols if col in display_df.columns]

            self.ui.print_table(
                display_df[final_cols],
                title=f"Recent Data & Indicators (Last {num_detail_rows})",
                float_format="{:.5f}",
                index=False,
            )

        elif not df_indicators.empty:  # Handle case where less than num_detail_rows exist
            self.ui.print_warning(
                f"Insufficient data ({len(df_indicators)} rows) to show detailed table of {num_detail_rows} rows. Showing all available."
            )
            display_df = df_indicators.copy()
            if isinstance(display_df.index, pd.DatetimeIndex):
                display_df.index = display_df.index.strftime("%m-%d %H:%M")
            display_df = display_df.reset_index()
            base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            indicator_cols = sorted([col for col in display_df.columns if col not in base_cols])
            final_cols = [col for col in base_cols + indicator_cols if col in display_df.columns]
            self.ui.print_table(
                display_df[final_cols],
                title="Recent Data & Indicators (All Available)",
                float_format="{:.5f}",
                index=False,
            )
        else:
            # This case should ideally be caught earlier if df_indicators is empty
            self.ui.print_warning("No data available for detailed indicator table.")

        await self.ui_wait_for_enter_async()

    async def settings_menu(self) -> None:
        """Handles the settings menu using async input."""
        settings_running = True
        while settings_running and self._running:
            if self._shutdown_event.is_set():
                break  # Check shutdown flag

            # Reload current config state for display
            current_config = self.config_manager.config
            colors = self.ui.colors  # Use current UI colors

            # Build options dynamically with current values
            options_map = {
                "1": ("Theme", current_config.get("theme")),
                "2": ("Default Symbol", current_config.get("default_symbol")),
                "3": ("Default Timeframe", current_config.get("default_timeframe")),
                "4": ("Default Order Type", current_config.get("default_order_type")),
                "5": ("Log Level", current_config.get("log_level")),
                "6": ("Pivot Period", current_config.get("pivot_period")),
                "7": ("Chart Points", current_config.get("chart_points")),
                "8": ("Chart Height", current_config.get("chart_height")),
                "9": ("Back to Main Menu", None),  # No current value needed
            }
            # Format options for display using current theme colors
            options_display = [
                f"{v[0]} [{colors['accent']}{v[1]}{colors['menu_option']}]" if v[1] is not None else v[0]
                for k, v in options_map.items()
            ]

            # Use async helper for menu display wrapped in another helper
            async def _get_settings_choice():
                loop = asyncio.get_running_loop()
                func_call = partial(
                    self.ui.display_menu, "Settings Configuration", options_display, "Select setting to change"
                )
                try:
                    # Run the blocking menu display in executor
                    return await loop.run_in_executor(None, func_call)
                except (EOFError, KeyboardInterrupt) as e:
                    raise e  # Re-raise to be caught below
                except Exception as e:
                    # Wrap other exceptions from the executor
                    raise RuntimeError(f"Failed to display settings menu: {e}") from e

            try:
                choice = await _get_settings_choice()
                if choice is None:  # Should not happen if menu handles errors, but check
                    logger.warning("Settings menu returned None unexpectedly.")
                    continue
            except (EOFError, KeyboardInterrupt) as e:
                logger.info(f"Settings menu interrupted ('{type(e).__name__}'), returning to main menu.")
                # Don't trigger shutdown here, just exit settings loop
                settings_running = False
                break  # Break inner while loop
            except RuntimeError as e:
                self.ui.print_error(str(e))
                await asyncio.sleep(1)
                continue  # Retry settings menu display

            if not self._running or self._shutdown_event.is_set():
                break  # Check global flags again after input

            # --- Handle Settings Choices ---
            needs_save = False
            try:
                if choice == "1":  # Change Theme
                    theme_input = await self.ui_get_input_async(
                        "Enter theme (dark/light)",
                        default=current_config.get("theme"),
                        validation_func=lambda t: t.lower() in ["dark", "light"] or "Must be 'dark' or 'light'",
                    )
                    if theme_input:
                        new_theme = theme_input.lower()
                        if new_theme != current_config.get("theme"):
                            self.config_manager.config["theme"] = new_theme
                            # Update colors immediately
                            self.config_manager.theme_colors = self.config_manager._setup_theme_colors()
                            self.ui.colors = self.config_manager.theme_colors
                            needs_save = True
                            self.ui.print_success(f"Theme changed to {new_theme}. Applied immediately.")
                        else:
                            self.ui.print_info("Theme unchanged.")
                    await asyncio.sleep(0.5)  # Short pause

                elif choice == "2":  # Set Default Symbol
                    symbol_input = await self.ui_get_input_async(
                        "Enter default symbol (e.g., BTC/USDT)",
                        default=current_config.get("default_symbol"),
                        validation_func=self._validate_symbol,
                    )
                    if symbol_input:
                        new_symbol = self._get_ccxt_symbol(symbol_input)
                        # Re-validate the formatted symbol before saving
                        validation_result = self._validate_symbol(new_symbol)
                        if validation_result is True:
                            if new_symbol != current_config.get("default_symbol"):
                                self.config_manager.config["default_symbol"] = new_symbol
                                needs_save = True
                                self.ui.print_success(f"Default symbol set to {new_symbol}.")
                            else:
                                self.ui.print_info("Default symbol unchanged.")
                        else:
                            self.ui.print_error(f"Validation failed for symbol '{new_symbol}': {validation_result}")
                    await asyncio.sleep(0.5)

                elif choice == "3":  # Set Default Timeframe
                    tf_input = await self.ui_get_input_async(
                        "Enter default timeframe (e.g., 1h)",
                        default=current_config.get("default_timeframe"),
                        validation_func=self._validate_timeframe,
                    )
                    if tf_input:
                        # Normalize timeframe
                        new_tf = tf_input.lower()
                        tf_map = {"d": "1d", "w": "1w", "m": "1M"}
                        if new_tf.upper() in tf_map:
                            new_tf = tf_map[new_tf.upper()]

                        if new_tf != current_config.get("default_timeframe"):
                            self.config_manager.config["default_timeframe"] = new_tf
                            needs_save = True
                            self.ui.print_success(f"Default timeframe set to {new_tf}.")
                        else:
                            self.ui.print_info("Default timeframe unchanged.")
                    await asyncio.sleep(0.5)

                elif choice == "4":  # Set Default Order Type
                    type_input = await self.ui_get_input_async(
                        "Enter default order type (Market/Limit)",
                        default=current_config.get("default_order_type"),
                        validation_func=self._validate_order_type,
                    )
                    if type_input:
                        new_type_cap = type_input.capitalize()  # Store capitalized version
                        if new_type_cap != current_config.get("default_order_type"):
                            self.config_manager.config["default_order_type"] = new_type_cap
                            needs_save = True
                            self.ui.print_success(f"Default order type set to {new_type_cap}.")
                        else:
                            self.ui.print_info("Default order type unchanged.")
                    await asyncio.sleep(0.5)

                elif choice == "5":  # Set Log Level
                    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    level_input = await self.ui_get_input_async(
                        f"Enter log level ({'/'.join(log_levels)})",
                        default=current_config.get("log_level"),
                        validation_func=lambda L: L.upper() in log_levels or f"Must be one of: {', '.join(log_levels)}",
                    )
                    if level_input:
                        new_level_upper = level_input.upper()
                        if new_level_upper != current_config.get("log_level"):
                            self.config_manager.config["log_level"] = new_level_upper
                            self.config_manager._apply_log_level()  # Apply immediately
                            needs_save = True
                            self.ui.print_success(f"Log level set to {new_level_upper}. Applied immediately.")
                            logger.info(f"Log level changed to {new_level_upper} via settings.")  # Log the change
                        else:
                            self.ui.print_info("Log level unchanged.")
                    await asyncio.sleep(0.5)

                elif choice == "6":  # Set Pivot Period
                    pivot_tf_input = await self.ui_get_input_async(
                        "Enter timeframe for pivot points (e.g., 1h, 1d, 1W)",
                        default=current_config.get("pivot_period"),
                        validation_func=self._validate_timeframe,
                    )
                    if pivot_tf_input:
                        # Normalize timeframe
                        new_pivot_tf = pivot_tf_input.lower()
                        tf_map = {"d": "1d", "w": "1w", "m": "1M"}
                        if new_pivot_tf.upper() in tf_map:
                            new_pivot_tf = tf_map[new_pivot_tf.upper()]

                        if new_pivot_tf != current_config.get("pivot_period"):
                            self.config_manager.config["pivot_period"] = new_pivot_tf
                            needs_save = True
                            self.ui.print_success(f"Pivot point calculation period set to {new_pivot_tf}.")
                        else:
                            self.ui.print_info("Pivot period unchanged.")
                    await asyncio.sleep(0.5)

                elif choice == "7":  # Set Chart Points

                    def points_validator(x):
                        return (isinstance(x, int) and 10 <= x <= 1000) or "Must be integer between 10 and 1000"

                    points_input = await self.ui_get_input_async(
                        "Enter number of points for chart (10-1000)",
                        default=current_config.get("chart_points"),
                        input_type=int,
                        validation_func=points_validator,
                    )
                    if points_input is not None:  # Input is guaranteed to be int by validation
                        if points_input != current_config.get("chart_points"):
                            self.config_manager.config["chart_points"] = points_input
                            needs_save = True
                            self.ui.print_success(f"Chart points set to {points_input}.")
                        else:
                            self.ui.print_info("Chart points unchanged.")
                    await asyncio.sleep(0.5)

                elif choice == "8":  # Set Chart Height

                    def height_validator(x):
                        return (isinstance(x, int) and 5 <= x <= 50) or "Must be integer between 5 and 50"

                    height_input = await self.ui_get_input_async(
                        "Enter height for ASCII chart (5-50)",
                        default=current_config.get("chart_height"),
                        input_type=int,
                        validation_func=height_validator,
                    )
                    if height_input is not None:
                        if height_input != current_config.get("chart_height"):
                            self.config_manager.config["chart_height"] = height_input
                            needs_save = True
                            self.ui.print_success(f"Chart height set to {height_input}.")
                        else:
                            self.ui.print_info("Chart height unchanged.")
                    await asyncio.sleep(0.5)

                elif choice == "9":  # Back
                    settings_running = False
                    # Save config if changes were made before exiting settings
                    if needs_save:
                        self.config_manager.save_config()
                        self.ui.print_info("Configuration saved.")
                        await asyncio.sleep(0.5)
                else:
                    # Should not happen due to menu validation
                    self.ui.print_error("Invalid settings choice received.")
                    await asyncio.sleep(1)

                # Save config after each successful change (or just before exiting)
                if needs_save and settings_running:  # Save if changed and not exiting yet
                    self.config_manager.save_config()
                    needs_save = False  # Reset flag after saving

            except (EOFError, KeyboardInterrupt) as e:
                # Should be caught by the _get_settings_choice wrapper, but handle defensively
                logger.warning(f"Interruption '{type(e).__name__}' during settings input. Exiting settings.")
                settings_running = False  # Exit settings loop
                if needs_save:
                    self.config_manager.save_config()  # Save pending changes on interrupt
                break
            except ValueError as e:  # Catch input conversion errors
                self.ui.print_error(f"Invalid input value: {e}")
                await asyncio.sleep(1)
            except Exception as e:  # Catch unexpected errors during setting update
                logger.error(f"Error handling setting choice '{choice}': {e}", exc_info=True)
                self.ui.print_error(f"An unexpected error occurred: {e}")
                if needs_save:
                    self.config_manager.save_config()  # Attempt save on error too
                await self.ui_wait_for_enter_async()


# --- Main Execution ---


def check_create_env_file() -> bool:
    """Checks for .env file and creates a default one if missing."""
    env_path = Path(".env")
    if not env_path.exists():
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("# Bybit API Credentials (replace with your actual keys)\n")
                f.write("# Get keys from Bybit website: Account -> API Management\n")
                f.write(
                    "# Ensure API key has permissions for Contract Trade (and Read-Only for Balance/Positions if separate)\n"
                )
                f.write("# For Unified Trading Account (UTA), ensure keys have appropriate permissions.\n")
                f.write("BYBIT_API_KEY=your_api_key_here\n")
                f.write("BYBIT_API_SECRET=your_api_secret_here\n\n")
                f.write("# Set to True to use Bybit's testnet environment\n")
                f.write("# Testnet URL: https://testnet.bybit.com (Requires separate testnet API keys)\n")
                f.write("TESTNET=False\n")
            return False  # Indicate that user action is required
        except OSError as e:
            logger.critical(f"Failed to create .env file: {e}")
            return False  # Indicate failure
    # File exists, check if placeholders are still present
    else:
        try:
            with open(env_path, encoding="utf-8") as f:
                content = f.read()
                if "your_api_key_here" in content or "your_api_secret_here" in content:
                    pass
                    # Allow script to continue but credentials check will likely fail later
        except OSError:
            pass
            # Proceed cautiously
    return True  # File exists (even if it contains placeholders)


async def main():
    """Main asynchronous function to initialize and run the trading terminal."""
    if not check_create_env_file():
        return 1  # Return non-zero exit code

    terminal = TradingTerminal()
    main_task = None
    exit_code = 0  # Default exit code
    try:
        # Create the main task using the helper to track it
        main_task = terminal._create_task(terminal.run(), name="MainTerminalRun")
        await main_task
        # Check the exit code set by shutdown if available
        exit_code = terminal._final_exit_code

    except asyncio.CancelledError:
        # This happens if the main_task itself is cancelled (e.g., during shutdown sequence initiated elsewhere)
        logger.info("Main terminal task was cancelled.")
        # Ensure cleanup runs if cancellation happened abruptly and shutdown wasn't fully completed
        if terminal and not terminal._shutdown_event.is_set():
            logger.warning("Main task cancelled but shutdown not fully complete. Forcing final shutdown steps.")
            # Use create_task to avoid awaiting shutdown within the cancel handler
            # Give it a moment, but don't block indefinitely
            shutdown_task = terminal._create_task(
                terminal.shutdown(signal="MainTaskCancel"), name="ForcedShutdownOnCancel"
            )
            try:
                await asyncio.wait_for(shutdown_task, timeout=2.0)
            except TimeoutError:
                logger.error("Forced shutdown task timed out.")
            except Exception as e:
                logger.error(f"Error during forced shutdown: {e}")
        exit_code = terminal._final_exit_code if terminal else 1  # Get code if set, otherwise assume error

    except KeyboardInterrupt:
        # This should ideally be caught by the signal handler within `run`,
        # but catch here as a fallback (e.g., interrupt during initial setup before handlers are active).
        logger.warning("KeyboardInterrupt caught directly in main async function.")
        if terminal and not terminal._shutdown_event.is_set():
            # Manually trigger shutdown if the handler didn't catch it or wasn't set yet
            # We need to run this shutdown; awaiting it might be tricky here.
            # Best effort: create task and give it time.
            shutdown_task = terminal._create_task(
                terminal.shutdown(signal="KeyboardInterrupt", exit_code=1), name="ForcedShutdownOnKBInterrupt"
            )
            try:
                await asyncio.wait_for(shutdown_task, timeout=2.0)
            except TimeoutError:
                logger.error("Forced shutdown task timed out on KBInterrupt.")
            except Exception as e:
                logger.error(f"Error during forced shutdown on KBInterrupt: {e}")
        exit_code = 1  # Signal abnormal termination

    except Exception as e:
        logger.critical(f"Critical unhandled error in main async function: {e}", exc_info=True)
        exit_code = 1  # Signal critical failure
        if terminal and not terminal._shutdown_event.is_set():
            logger.info("Attempting emergency shutdown after critical error...")
            # Best effort shutdown
            shutdown_task = terminal._create_task(
                terminal.shutdown(exit_code=1, signal="CriticalError"), name="ForcedShutdownOnError"
            )
            try:
                await asyncio.wait_for(shutdown_task, timeout=2.0)
            except TimeoutError:
                logger.error("Emergency shutdown task timed out.")
            except Exception as se:
                logger.error(f"Error during emergency shutdown: {se}")

    finally:
        # Final cleanup check, although shutdown should handle most of it
        if terminal and terminal.exchange_client and terminal.exchange_client._initialized:
            # Check if client still exists and wasn't closed by shutdown already
            if terminal.exchange_client.exchange:
                logger.info("Performing final check for client connection closure.")
                await terminal.exchange_client.close()  # Ensure closed

        logger.info("Main async function finished.")
        # Return the determined exit code
        return exit_code


if __name__ == "__main__":
    # Check Python version
    # Allow running on < 3.8, but features might be limited or fail

    final_exit_code = 0
    try:
        # Run the main async function using asyncio.run() which returns the value from main()
        final_exit_code = asyncio.run(main())
        # asyncio.run() handles loop creation/closing.

    except KeyboardInterrupt:
        # Catch KeyboardInterrupt during asyncio.run() startup or final shutdown phase (less likely but possible)
        # Use a different logger name or just print, as terminal logger might be shut down
        logging.getLogger("MainRunner").warning("KeyboardInterrupt caught outside main coroutine (during asyncio.run).")
        final_exit_code = 1  # Signal abnormal termination
    except Exception as e:
        # Catch truly fatal errors during startup or final shutdown managed by asyncio.run()
        logging.getLogger("MainRunner").critical(f"Fatal error outside main coroutine: {e}", exc_info=True)
        final_exit_code = 1  # Signal critical failure
    finally:
        # Ensure logging is flushed and handlers closed before exiting the process
        logging.shutdown()
        # Final message indicating the script is ending
        sys.exit(final_exit_code)
