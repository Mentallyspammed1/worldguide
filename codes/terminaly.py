Okay, let's go through the provided code, correct observed issues, and implement the enhancements mentioned in the description.

The primary focus will be on:
1.  **Async Input Fix:** Ensuring `input()` calls don't block the asyncio loop by running them in an executor.
2.  **V5 API Compatibility:** Adding `category='linear'` and `accountType='CONTRACT'` where needed for Bybit V5.
3.  **Error Handling:** Catching specific CCXT exceptions and providing user-friendly feedback.
4.  **Configuration:** Robust loading, merging, and normalization of settings.
5.  **UI/Table Handling:** Fixing color display issues in tables (tabulate doesn't like ANSI codes).
6.  **Validation:** Implementing input validation functions and integrating them.
7.  **Shutdown:** Ensuring graceful shutdown via signals and menu.
8.  **TA Enhancements:** Ensuring TA calculation is robust and display is clear.

I will go section by section, making the necessary changes.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bybit Futures Terminal - v2.9 - Pyrmethus Enhanced Edition
Original Author: Mentallyspammed1 (Enhanced by AI & Pyrmethus)
Last Updated: 2025-05-01 (Pyrmethus Refinement Pass 2) - Enhanced based on review, fixing observed errors, improving robustness, and refining async/UI interaction.

Description:
A command-line interface for interacting with Bybit Futures (USDT Perpetual).
This enhanced version focuses on:
- Unified API interaction via the CCXT library for consistency and robustness (V5 focused).
- Improved error handling and user feedback, including specific CCXT exceptions.
- Correct asynchronous operations using standard asyncio with non-blocking input handling via executor.
- Configuration management via JSON and .env files with safe merging and normalization.
- Enhanced technical analysis: MAs, RSI, MACD, BBands, STOCH, ATR, Pivots, Trend Summary.
- Trailing Stop orders (percentage-based), Price Check, Open Positions/Orders view.
- Clear, color-coded terminal UI using Colorama and improved table formatting (tabulate fallback for colors).
- Robust ASCII charting for price trends with color.
- Fixed async input handling using asyncio's run_in_executor and functools.partial.
- Graceful shutdown handling via signals (SIGINT, SIGTERM) and menu option.
- Automatic .env file creation on first run with instructions.
- Robust input validation for symbols, timeframes, order types, quantities, etc., with suggestions.
- Dynamic menu options based on API connection status (displays disabled options).
- Fixes for observed errors (DataFrame boolean ambiguity, V5 balance fetch, input validation, color handling in tables).
- Improved V5 API compliance (category/accountType params).
- Enhanced credential checking and .env file guidance.
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from functools import wraps, partial # partial is used for async input handling
from pathlib import Path
import signal  # For graceful shutdown
import re # For timeframe and symbol validation

# Async & CCXT
import ccxt.async_support as ccxt

# Data & Analysis
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    # Import error message moved to where TA is used, as Fore is not yet initialized here
    ta = None # Set ta to None if not installed

# Configuration & Environment
from dotenv import load_dotenv

# UI & Utilities
from colorama import init, Fore, Back, Style
import asciichartpy as asciichart
try:
    # Used for better table formatting in print_table
    from tabulate import tabulate
except ImportError:
    # Import error message moved to print_table, as Fore is not yet initialized here
    tabulate = None # Set to None if not installed


# --- Initial Setup ---

# Initialize colorama for cross-platform terminal colors FIRST
# autoreset=True ensures styles are reset after each print automatically
init(autoreset=True)

# Configure logging
log_file = 'terminal.log'
# Ensure log file directory exists (useful if run from different locations)
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
# Configure logging format and handlers. Level is set by ConfigManager later.
logging.basicConfig(
    level=logging.INFO, # Default level, will be overridden by config later
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'), # Specify encoding
        logging.StreamHandler(sys.stdout) # Log to stdout as well
    ]
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
        "default_symbol": "BTC/USDT", # Use CCXT standard format
        "default_timeframe": "1h",    # Use CCXT standard format (lowercase)
        "default_order_type": "Limit", # Capitalized
        "connection_timeout_ms": 30000,
        "order_history_limit": 50, # Note: Not actively used yet, kept for future expansion
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
        "pivot_period": "1d" # Timeframe for pivot point calculation (use lowercase CCXT standard)
    }

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config: Dict = self._load_config()
        # Apply log level early based on loaded/default config *before* setting up theme colors
        self._apply_log_level()
        # Now that log level is set, we can safely use logger.info with colors
        self.theme_colors: Dict = self._setup_theme_colors() # Setup theme based on loaded/default config
        logger.info(f"{self.theme_colors['primary']}# Configuration spell cast from '{self.config_path}'{self.theme_colors['reset']}")

    def _normalize_config_values(self, config_dict: Dict) -> Dict:
        """Normalizes specific configuration values to standard formats and types."""
        # Create a copy to avoid modifying the input dict directly
        normalized_dict = config_dict.copy()

        # Use defaults from DEFAULT_CONFIG if keys are missing in config_dict
        # String Normalization
        normalized_dict['default_timeframe'] = str(normalized_dict.get('default_timeframe', self.DEFAULT_CONFIG['default_timeframe'])).strip().lower()
        normalized_dict['pivot_period'] = str(normalized_dict.get('pivot_period', self.DEFAULT_CONFIG['pivot_period'])).strip().lower()
        normalized_dict['default_symbol'] = str(normalized_dict.get('default_symbol', self.DEFAULT_CONFIG['default_symbol'])).strip().upper()
        normalized_dict['default_order_type'] = str(normalized_dict.get('default_order_type', self.DEFAULT_CONFIG['default_order_type'])).strip().capitalize()
        normalized_dict['log_level'] = str(normalized_dict.get('log_level', self.DEFAULT_CONFIG['log_level'])).strip().upper()
        normalized_dict['theme'] = str(normalized_dict.get('theme', self.DEFAULT_CONFIG['theme'])).strip().lower()

        # Numeric Normalization - Use default if conversion fails
        numeric_keys = [
            ('connection_timeout_ms', int),
            ('order_history_limit', int),
            ('chart_height', int),
            ('chart_points', int)
        ]
        for key, target_type in numeric_keys:
             default_value = self.DEFAULT_CONFIG[key]
             current_value = normalized_dict.get(key, default_value)
             try:
                 normalized_dict[key] = target_type(current_value)
             except (ValueError, TypeError):
                  logger.warning(f"Invalid value '{current_value}' for config key '{key}'. Expected {target_type.__name__}. Using default: {default_value}")
                  normalized_dict[key] = default_value

        # Indicator Periods Normalization - Ensure numeric, use default if conversion fails
        if 'indicator_periods' in normalized_dict and isinstance(normalized_dict['indicator_periods'], dict):
            default_periods = self.DEFAULT_CONFIG['indicator_periods']
            # Ensure nested dict exists if missing
            if not isinstance(normalized_dict['indicator_periods'], dict):
                 logger.warning(f"Config key 'indicator_periods' is not a dictionary. Using default indicator periods.")
                 normalized_dict['indicator_periods'] = {} # Replace with empty dict if invalid type

            for key, default_value in default_periods.items():
                 current_value = normalized_dict['indicator_periods'].get(key, default_value)
                 target_type = type(default_value) # Infer type from default
                 try:
                     # Attempt conversion to the default value's type
                     normalized_dict['indicator_periods'][key] = target_type(current_value)
                 except (ValueError, TypeError):
                      logger.warning(f"Invalid value '{current_value}' for indicator period '{key}'. Expected {target_type.__name__}. Using default: {default_value}")
                      normalized_dict['indicator_periods'][key] = default_value
        else:
             # If 'indicator_periods' key is missing or not a dict, use the default dict
             logger.warning(f"Config key 'indicator_periods' is missing or invalid. Using default indicator periods.")
             normalized_dict['indicator_periods'] = self.DEFAULT_CONFIG['indicator_periods'].copy()

        return normalized_dict

    def _load_config(self) -> Dict:
        """Loads configuration from JSON file, merges with defaults, normalizes, or creates default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Ensure nested defaults are present if missing in loaded config
                    merged_config = self._deep_merge_dicts(self.DEFAULT_CONFIG.copy(), loaded_config)
                    normalized_config = self._normalize_config_values(merged_config)
                    return normalized_config
            except json.JSONDecodeError:
                # Use print here as logger might not be fully configured with colors yet
                print(f"{Fore.RED}Error decoding JSON from '{self.config_path}'. Conjuring default config.{Style.RESET_ALL}", file=sys.stderr)
                # Create default file and return normalized default
                return self._create_default_config()
            except Exception as e:
                print(f"{Fore.RED}Failed to load config from '{self.config_path}': {e}. Conjuring default config.{Style.RESET_ALL}", file=sys.stderr)
                # Return normalized default in memory
                return self._normalize_config_values(self.DEFAULT_CONFIG.copy())
        else:
            print(f"{Fore.YELLOW}'{self.config_path}' not found. Conjuring default configuration.{Style.RESET_ALL}", file=sys.stderr)
            # Create file and return normalized default
            return self._create_default_config()

    def _deep_merge_dicts(self, base: Dict, update: Dict) -> Dict:
        """Recursively merges update dict into base dict."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                # Recurse into nested dictionaries
                base[key] = self._deep_merge_dicts(base[key], value)
            else:
                # Overwrite non-dict values or add new keys/dicts
                base[key] = value
        return base

    def _create_default_config(self) -> Dict:
        """Creates a default configuration file and returns the normalized default config."""
        normalized_default_config = self._normalize_config_values(self.DEFAULT_CONFIG.copy())
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding='utf-8') as f:
                json.dump(normalized_default_config, f, indent=4, ensure_ascii=False)
            # Use print here as logger might not be fully configured with colors yet
            print(f"{Fore.GREEN}Default configuration scroll inscribed at '{self.config_path}'{Style.RESET_ALL}")
            return normalized_default_config
        except IOError as e:
            print(f"{Fore.RED}Error inscribing default config scroll: {e}. Using in-memory default.{Style.RESET_ALL}", file=sys.stderr)
            # Return the config anyway, it exists in memory
            return normalized_default_config
        except Exception as e:
             print(f"{Fore.RED}Unexpected error creating default configuration: {e}{Style.RESET_ALL}", file=sys.stderr)
             # Return the config anyway, it exists in memory
             return normalized_default_config


    def save_config(self):
        """Saves the current configuration to the file after normalization."""
        try:
            # Ensure current config is normalized before saving
            # This also handles potential type issues introduced by user input in settings
            normalized_config_to_save = self._normalize_config_values(self.config.copy())
            self.config = normalized_config_to_save # Update in-memory config to normalized version

            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f"{self.theme_colors['success']}Configuration scroll updated at '{self.config_path}'{self.theme_colors['reset']}")
        except IOError as e:
            logger.error(f"{self.theme_colors['error']}Error saving configuration scroll: {e}{self.theme_colors['reset']}")
        except Exception as e:
             logger.error(f"{self.theme_colors['error']}Unexpected error saving configuration: {e}{self.theme_colors['reset']}", exc_info=True)

    def _apply_log_level(self):
        """Applies the log level from the configuration."""
        # Get level string from config, default to INFO if missing or invalid type
        log_level_str = str(self.config.get("log_level", "INFO")).strip().upper()
        log_level = getattr(logging, log_level_str, logging.INFO) # Fallback to INFO if string invalid

        # Set levels for root logger and our specific logger
        # Setting root logger level is often necessary for handlers to respect the level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        logger.setLevel(log_level)

        # Ensure handlers also respect the new level
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

        # Log the level change using the logger itself
        logger.info(f"{self.theme_colors.get('info', Fore.BLUE)}# Log level attuned to {log_level_str}{self.theme_colors.get('reset', Style.RESET_ALL)}")


    def _setup_theme_colors(self) -> Dict:
        """Sets up theme colors based on the configuration."""
        # Get theme string from config, default to dark if missing or invalid type
        theme = str(self.config.get("theme", "dark")).strip().lower()

        if theme == "dark":
            # Dark theme with mystical hues
            return {
                'primary': Fore.CYAN, 'secondary': Fore.MAGENTA, 'accent': Fore.YELLOW,
                'error': Fore.RED + Style.BRIGHT, 'success': Fore.GREEN + Style.BRIGHT,
                'warning': Fore.YELLOW, 'info': Fore.BLUE + Style.BRIGHT,
                'title': Fore.CYAN + Style.BRIGHT, 'menu_option': Fore.WHITE,
                'menu_highlight': Fore.YELLOW + Style.BRIGHT,
                'input_prompt': Fore.MAGENTA + Style.BRIGHT,
                'table_header': Fore.CYAN + Style.BRIGHT,
                'positive': Fore.GREEN, 'negative': Fore.RED, 'neutral': Fore.WHITE,
                'dim': Style.DIM, 'reset': Style.RESET_ALL,
            }
        else: # Light theme (Adjusted for better visibility on light backgrounds)
             return {
                'primary': Fore.BLUE, 'secondary': Fore.GREEN, 'accent': Fore.MAGENTA,
                'error': Fore.RED + Style.BRIGHT, 'success': Fore.GREEN + Style.BRIGHT,
                'warning': Fore.YELLOW + Style.BRIGHT, # Brighter yellow stands out
                'info': Fore.CYAN + Style.BRIGHT,
                'title': Fore.BLUE + Style.BRIGHT, 'menu_option': Fore.BLACK, # Black text for light background
                'menu_highlight': Fore.BLUE + Style.BRIGHT,
                'input_prompt': Fore.MAGENTA + Style.BRIGHT,
                'table_header': Fore.BLUE + Style.BRIGHT,
                'positive': Fore.GREEN, 'negative': Fore.RED, 'neutral': Fore.BLACK, # Black neutral
                'dim': Style.DIM, 'reset': Style.RESET_ALL, # Keep dim/reset standard
            }

# --- CCXT Exchange Client ---

class BybitFuturesCCXTClient:
    """
    Client for interacting with Bybit Futures via CCXT.
    Handles initialization, context management, and core API calls (V5 focus).
    """
    def __init__(self, credentials: APICredentials, config: Dict):
        self.credentials = credentials
        self.config = config
        self.exchange: Optional[ccxt.bybit] = None
        self._initialized = False
        self._markets_loaded = False
        # Get colors from a temporary UI instance just for logging/messages here
        self._colors = ConfigManager().theme_colors


    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize only if not already initialized
        if not self._initialized: await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Sets up the CCXT Bybit exchange instance, loads markets, handles V5 specifics."""
        if self._initialized:
            logger.debug("CCXT client already initialized.")
            return

        logger.info(f"{self._colors['primary']}# Summoning connection to Bybit via CCXT...{self._colors['reset']}")
        exchange_config = {
            'apiKey': self.credentials.api_key,
            'secret': self.credentials.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap', # Explicitly target perpetual swaps
                'adjustForTimeDifference': True, # Sync time with exchange
                # V5 API requires 'category' often. While defaultType helps,
                # we will explicitly add 'category': 'linear' in API call params where needed for robustness.
                # 'defaultMarginMode': 'isolated', # Or 'cross'. Set if needed globally, but usually position-specific.
            },
            'timeout': self.config.get("connection_timeout_ms", 30000), # Use config timeout
        }

        self.exchange = ccxt.bybit(exchange_config) # Initialize first

        if self.credentials.testnet:
            logger.warning(f"{self._colors['warning']}# Engaging Bybit TESTNET dimension.{self._colors['reset']}")
            # Prefer set_sandbox_mode for robustness if available
            if hasattr(self.exchange, 'set_sandbox_mode') and callable(self.exchange.set_sandbox_mode):
                try:
                    self.exchange.set_sandbox_mode(True)
                    logger.info(f"{self._colors['info']}Using exchange.set_sandbox_mode(True) for Testnet.{self._colors['reset']}")
                except Exception as e:
                    logger.error(f"{self._colors['error']}Error calling set_sandbox_mode: {e}. Testnet may not function correctly.{self._colors['reset']}")
                    # If set_sandbox_mode exists but fails, direct URL might be the only fallback (though less ideal)
                    logger.warning(f"{self._colors['warning']}Falling back to manually setting testnet API URL due to set_sandbox_mode error.{self._colors['reset']}")
                    self.exchange.urls['api'] = self.exchange.urls['test']
            else:
                # If method doesn't exist, fallback to setting URL
                logger.warning(f"{self._colors['warning']}exchange.set_sandbox_mode() not available. Manually setting testnet API URL.{self._colors['reset']}")
                self.exchange.urls['api'] = self.exchange.urls['test']

        try:
            logger.info(f"{self._colors['info']}# Loading market contracts...{self._colors['reset']}")
            # Load markets with retries in case of temporary network issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Use 'reload=True' to ensure fresh market data on initialization
                    await self.exchange.load_markets(reload=True)
                    if self.exchange.markets:
                        break # Success, markets loaded
                    else:
                         logger.warning(f"{self._colors['warning']}Market loading attempt {attempt + 1} of {max_retries} returned no markets. Retrying...{self._colors['reset']}")
                         # Use exponential backoff with jitter
                         await asyncio.sleep(1.5 ** attempt + random.random() * 0.5) # Add random jitter
                except (ccxt.NetworkError, asyncio.TimeoutError, ccxt.DDoSProtection) as net_err:
                     if attempt < max_retries - 1:
                         logger.warning(f"{self._colors['warning']}Network error loading markets (Attempt {attempt + 1}/{max_retries}): {net_err}. Retrying...{self._colors['reset']}")
                         await asyncio.sleep(1.5 ** attempt + random.random() * 0.5)
                     else:
                         logger.error(f"{self._colors['error']}Failed to load markets after {max_retries} attempts due to network errors.{self._colors['reset']}")
                         raise net_err # Re-raise the last network error
                except ccxt.ExchangeError as ex_err:
                     # Catch specific exchange errors during market load (e.g., rate limit, invalid params)
                     logger.error(f"{self._colors['error']}Exchange error loading markets (Attempt {attempt + 1}/{max_retries}): {ex_err}. {'Retrying...' if attempt < max_retries - 1 else ''}{self._colors['reset']}")
                     if attempt < max_retries - 1:
                          await asyncio.sleep(1.5 ** attempt + random.random() * 0.5)
                     else:
                          raise ex_err

            # After loop, check if markets were actually loaded
            if not self.exchange.markets:
                 logger.error(f"{self._colors['error']}Failed to load markets from Bybit after retries. Check connection and API permissions.{self._colors['reset']}")
                 # Raise a ConnectionError to be caught by the caller (TradingTerminal.initialize)
                 raise ConnectionError("Market loading failed - no markets returned after retries")
            else:
                 # Filter for active USDT perpetual swaps (linear) for informative log
                 usdt_swap_markets = {
                     k: v for k, v in self.exchange.markets.items()
                     if v.get('swap') and v.get('linear') and v.get('quote') == 'USDT' and v.get('active') # Ensure market is active
                 }
                 if not usdt_swap_markets:
                     logger.warning(f"{self._colors['warning']}No ACTIVE USDT Perpetual Swap markets found after loading. Check API key permissions or market availability.{self._colors['reset']}")
                 else:
                     logger.info(f"{self._colors['success']}Loaded {len(self.exchange.markets)} total markets, found {len(usdt_swap_markets)} active USDT Perpetual Swaps.{self._colors['reset']}")
                 self._markets_loaded = True

            # Optional: Verify connectivity with a simple, low-impact call after loading markets
            # await self.exchange.fetch_time() # Can add latency but confirms API communication
            # logger.info("Connectivity test successful.")

            self._initialized = True
            logger.info(f"{self._colors['success']}CCXT Bybit Futures client materialized successfully for {'Testnet' if self.credentials.testnet else 'Mainnet'}.{self._colors['reset']}")

        except ccxt.AuthenticationError as e:
            logger.error(f"{self._colors['error']}CCXT Authentication Sigil Rejected: Invalid API keys or permissions. Check .env file and Bybit API settings. Ensure key has CONTRACT TRADE permissions. {e}{self._colors['reset']}", exc_info=False) # Less verbose logging for known auth errors
            await self.close() # Ensure cleanup on failure
            raise ConnectionError("CCXT Authentication Failed") from e # Re-raise as ConnectionError for terminal to catch
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, asyncio.TimeoutError, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Network Rift: Could not connect to Bybit. Check internet and exchange status. {e}{self._colors['reset']}", exc_info=True)
            await self.close()
            raise ConnectionError("CCXT Network/Connection Failed") from e # Re-raise as ConnectionError
        except ccxt.ExchangeError as e: # Catch other exchange-specific errors during init/market load
            logger.error(f"{self._colors['error']}CCXT Exchange Anomaly during initialization: {e}{self._colors['reset']}", exc_info=True)
            await self.close()
            raise ConnectionError("CCXT Exchange Initialization Failed") from e # Re-raise as ConnectionError
        except Exception as e: # Catch any other unexpected error
            logger.error(f"{self._colors['error']}An unexpected vortex occurred initializing CCXT: {e}{self._colors['reset']}", exc_info=True)
            await self.close()
            raise ConnectionError("Unexpected CCXT Initialization Error") from e # Re-raise as ConnectionError

    async def close(self):
        """Closes the underlying CCXT exchange connection gracefully and resets state."""
        # Use a flag to prevent concurrent close calls if possible, although CCXT close should be idempotent
        if self.exchange:
            # Check if 'close' method exists and is callable
            if hasattr(self.exchange, 'close') and callable(self.exchange.close):
                logger.info(f"{self._colors['primary']}# Banishing CCXT Bybit client connection...{self._colors['reset']}")
                try:
                    await self.exchange.close()
                    logger.info(f"{self._colors['success']}CCXT connection banished.{self._colors['reset']}")
                except Exception as e:
                    # Log error but proceed with cleanup
                    logger.error(f"{self._colors['error']}Error banishing CCXT connection: {e}{self._colors['reset']}", exc_info=False)
            else:
                 # Log if close method is not available (unlikely for modern ccxt but safe)
                 logger.debug("Exchange object does not have a callable 'close' method.")
        # Reset state regardless of whether close succeeded or was needed
        self.exchange = None
        self._initialized = False
        self._markets_loaded = False
        logger.debug("Client state reset.")

    def _check_initialized(self):
        """Raises a ConnectionError if the client is not properly initialized or markets not loaded."""
        # Use UI colors from a temporary instance as _colors might not be fully reliable here
        colors = ConfigManager().theme_colors
        if not self.exchange or not self._initialized or not self._markets_loaded:
            # Provide a more specific error message based on state
            if not self.credentials:
                reason = "API credentials not loaded."
            elif not self.exchange:
                 reason = "Exchange object not created (init failed early)."
            elif not self._initialized:
                 reason = "Initialization sequence did not complete successfully (e.g., auth/connection failure)."
            elif not self._markets_loaded:
                 reason = "Market data failed to load."
            else:
                 reason = "Client is not ready." # Generic fallback
            logger.error(f"{colors['error']}Client check failed: {reason}{colors['reset']}")
            raise ConnectionError(f"CCXT client is not ready: {reason}")

    async def fetch_balance(self) -> Dict:
        """Fetches account balance (USDT Futures) for V5 CONTRACT accounts."""
        self._check_initialized()
        logger.debug(f"{self._colors['info']}# Fetching arcane balance energies (CONTRACT account)...{self._colors['reset']}")
        try:
            # Bybit V5 Unified/Contract requires accountType param.
            # For USDT perpetuals (Linear Swap), the account type is 'CONTRACT'.
            params = {'accountType': 'CONTRACT'}
            balance = await self.exchange.fetch_balance(params=params)
            logger.debug(f"Raw balance data received (Account Type: CONTRACT): {balance}")

            # CCXT aims for standard structure: balance['USDT'] = {'free': ..., 'used': ..., 'total': ...}
            # Check standard structure first
            if 'USDT' in balance and isinstance(balance['USDT'], dict):
                usdt_balance = balance['USDT'].copy() # Work on a copy
                # Ensure info is present for detailed PNL extraction later if needed
                # Prefer the 'info' nested under 'USDT' if CCXT provides it, otherwise top-level 'info'
                usdt_balance_info = usdt_balance.get('info', balance.get('info', {}))
                usdt_balance['info'] = usdt_balance_info

                # Try to populate PNL fields if missing in standard keys but present in raw 'info'
                # This structure is based on Bybit V5 response for accountType=CONTRACT
                if 'info' in usdt_balance and isinstance(usdt_balance['info'], dict):
                     raw_result = usdt_balance['info'].get('result', {})
                     if isinstance(raw_result, dict):
                         raw_list = raw_result.get('list', [])
                         # Find the USDT asset within the list for the CONTRACT account
                         asset_info = None
                         if isinstance(raw_list, list):
                             for item in raw_list:
                                 # Check for both 'CONTRACT' and 'UNIFIED' account types as keys might be for either
                                 # Prioritize 'CONTRACT' if testnet is off, otherwise check both
                                 is_contract = item.get('accountType') == 'CONTRACT'
                                 is_unified = item.get('accountType') == 'UNIFIED'
                                 if (is_contract or (self.credentials.testnet and is_unified)) and item.get('coin') == 'USDT':
                                     asset_info = item
                                     break
                         if asset_info:
                             # Populate standard PNL fields if they are None, using V5 keys from asset_info
                             # Use float conversion with default 0 if key is missing or value is None/empty string
                             usdt_balance['unrealizedPnl'] = float(asset_info.get('unrealisedPnl', usdt_balance.get('unrealizedPnl', 0)))
                             usdt_balance['cumRealisedPnl'] = float(asset_info.get('cumRealisedPnl', usdt_balance.get('cumRealisedPnl', 0)))

                             # Populate standard free/used/total from V5 info if needed/available
                             usdt_balance['free'] = float(asset_info.get('availableToWithdraw', usdt_balance.get('free', 0)))
                             usdt_balance['used'] = float(asset_info.get('usedMargin', usdt_balance.get('used', 0)))
                             usdt_balance['total'] = float(asset_info.get('equity', usdt_balance.get('total', 0)))

                             # Add other useful V5 fields if needed, maybe prefixed like 'v5_equity'
                             usdt_balance['v5_equity'] = float(asset_info.get('equity', 0))
                             usdt_balance['v5_walletBalance'] = float(asset_info.get('walletBalance', 0))
                             usdt_balance['v5_availableToWithdraw'] = float(asset_info.get('availableToWithdraw', 0))

                logger.debug(f"Final standardized USDT balance (after info parse): {usdt_balance}")
                return usdt_balance

            # Fallback: If 'USDT' key missing or not a dict, try parsing directly from 'info'
            logger.warning(f"{self._colors['warning']}Standard 'USDT' balance structure not found. Attempting direct parse from 'info'.{self._colors['reset']}")
            if 'info' in balance and isinstance(balance['info'], dict) and 'result' in balance['info'] and isinstance(balance['info']['result'], dict) and 'list' in balance['info']['result']:
                 raw_list = balance['info']['result']['list']
                 if isinstance(raw_list, list):
                     for asset_info in raw_list:
                         # Find the USDT entry specifically for the CONTRACT account
                         is_contract = asset_info.get('accountType') == 'CONTRACT'
                         is_unified = asset_info.get('accountType') == 'UNIFIED'
                         if (is_contract or (self.credentials.testnet and is_unified)) and asset_info.get('coin') == 'USDT':
                             logger.debug(f"Parsing V5 balance structure directly from info: {asset_info}")
                             # Map Bybit keys to CCXT-like standard keys carefully
                             # Use float conversion with default 0 if key is missing or value is None/empty string
                             parsed_balance = {
                                 'free': float(asset_info.get('availableToWithdraw', 0)), # 'availableToWithdraw' seems the closest to 'free'
                                 'used': float(asset_info.get('usedMargin', 0)), # This might be order margin + position margin
                                 'total': float(asset_info.get('equity', 0)), # Equity is usually the most relevant 'total'
                                 'unrealizedPnl': float(asset_info.get('unrealisedPnl', 0)),
                                 'cumRealisedPnl': float(asset_info.get('cumRealisedPnl', 0)),
                                 'info': asset_info # Keep raw info
                             }
                             logger.info(f"{self._colors['warning']}Successfully parsed V5 balance directly from info.{self._colors['reset']}")
                             return parsed_balance

            logger.error(f"{self._colors['error']}Could not find or parse USDT CONTRACT balance data. Raw response keys: {list(balance.keys())}{self._colors['reset']}")
            # Return an empty dict or minimal structure if completely unparsable
            return {}

        except ccxt.BadRequest as e:
            # Catch the specific error observed: {"retCode":10001,"retMsg":"accountType only support UNIFIED."...}
            # This indicates the user might have a Unified account but is using API keys restricted to it,
            # while the terminal is requesting CONTRACT account balance using `accountType: 'CONTRACT'`.
            # Or the key simply doesn't have permissions for the CONTRACT account type balance.
             logger.error(f"{self._colors['error']}CCXT BadRequest fetching balance: {e}. This likely means the API key is for a UNIFIED account, or lacks permissions for CONTRACT account balance. Ensure the API key is configured correctly for the desired account type.{self._colors['reset']}", exc_info=False) # Less verbose logging for this specific known issue
             raise ConnectionError("Account type mismatch or insufficient permissions for balance fetch") from e # Re-raise as ConnectionError
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
             logger.error(f"{self._colors['error']}CCXT Auth/Permission Error fetching balance: {e}. Check API key validity and permissions (Read-Only for Wallet/Account needed).{self._colors['reset']}", exc_info=True)
             raise # Re-raise specific CCXT error
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Error fetching balance: {e}{self._colors['reset']}", exc_info=True)
            raise # Re-raise specific CCXT error
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected error fetching balance: {e}{self._colors['reset']}", exc_info=True)
            raise # Re-raise general error


    async def place_order(self, symbol: str, side: str, order_type: str, amount: float, price: Optional[float] = None, params: Dict = {}) -> Dict:
        """Places an order using CCXT, handling common exceptions and V5 params."""
        self._check_initialized()

        # Ensure V5 category param is present for USDT perpetuals
        # CCXT defaultType='swap' should handle this, but explicit params adds robustness
        if 'category' not in params:
            params['category'] = 'linear'

        # Normalize inputs just in case (though validation should handle this)
        symbol = symbol.upper()
        side = side.lower()
        order_type = order_type.lower() # CCXT expects lowercase type

        # Ensure price is None for market orders, required for limit
        if order_type == 'market' and price is not None:
             logger.warning(f"{self._colors['warning']}Price ({price}) provided for Market order, ignoring price.{self._colors['reset']}")
             price = None
        elif order_type == 'limit' and price is None:
             logger.error(f"{self._colors['error']}Limit order requires a price, but none was provided.{self._colors['reset']}")
             raise ccxt.InvalidOrder("Limit order requires a price.")

        logger.info(f"{self._colors['primary']}# Weaving {order_type.capitalize()} {side.upper()} order: {amount} {symbol} @ {price if price is not None else 'Market'} with params: {params}{self._colors['reset']}")
        try:
            # CCXT standard params: stopLossPrice, takeProfitPrice, trailingPercent, reduceOnly
            # These should map to Bybit's 'stopLoss', 'takeProfit', 'trailingStop', 'reduceOnly' via CCXT.
            # Bybit V5 often expects trigger prices as strings, CCXT should handle conversion.
            # Trailing Stop: CCXT 'trailingPercent' expects float. Bybit V5 expects 'trailingStop' as string (e.g., "0.5").
            # Let's rely on CCXT's mapping for 'trailingPercent'. If it fails, we might need to manually set 'trailingStop' in params.

            order = await self.exchange.create_order(symbol, order_type, side, amount, price, params)
            logger.info(f"{self._colors['success']}Order weaving successful: ID {order.get('id')}{self._colors['reset']}")
            # Log the full returned order structure for debugging
            logger.debug(f"Full order response: {order}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"{self._colors['error']}Order weaving failed: Insufficient magical essence (funds). Check available balance and margin requirements. {e}{self._colors['reset']}", exc_info=False)
            raise
        except ccxt.InvalidOrder as e:
            # Provide more context if possible by parsing the error message
            err_msg = str(e) # Keep case for potentially useful API messages
            err_msg_lower = err_msg.lower()
            if "order quantity" in err_msg_lower or "order qty" in err_msg_lower or "size" in err_msg_lower or "lot size" in err_msg_lower:
                 logger.error(f"{self._colors['error']}Order weaving failed: Invalid quantity. Check min/max order size and step size for {symbol}. {err_msg}{self._colors['reset']}", exc_info=False)
            elif "order price" in err_msg_lower or "price" in err_msg_lower or "tick size" in err_msg_lower:
                 logger.error(f"{self._colors['error']}Order weaving failed: Invalid price. Check price limits, tick size, or market conditions for {order_type} orders. {err_msg}{self._colors['reset']}", exc_info=False)
            elif "margin" in err_msg_lower or "leverage" in err_msg_lower:
                 logger.error(f"{self._colors['error']}Order weaving failed: Margin issue. Check available margin, leverage setting, and potential position size limits. {err_msg}{self._colors['reset']}", exc_info=False)
            elif "reduce-only" in err_msg_lower or "reduceonly" in err_msg_lower or "position idx" in err_msg_lower: # Bybit might mention position idx with reduceOnly
                 logger.error(f"{self._colors['error']}Order weaving failed: Reduce-only conflict or position mode issue. Cannot increase position size with reduce-only order. {err_msg}{self._colors['reset']}", exc_info=False)
            elif "trigger" in err_msg_lower or "stop loss" in err_msg_lower or "take profit" in err_msg_lower or "sl/tp" in err_msg_lower:
                 logger.error(f"{self._colors['error']}Order weaving failed: Invalid TP/SL parameters. Check trigger prices relative to current price/side and trigger type settings on Bybit. {err_msg}{self._colors['reset']}", exc_info=False)
            elif "trailing stop" in err_msg_lower or "trailingstop" in err_msg_lower:
                  logger.error(f"{self._colors['error']}Order weaving failed: Invalid Trailing Stop parameters. Check percentage value (e.g., 0.1 to 10), activation price (if applicable), or if feature is enabled. {err_msg}{self._colors['reset']}", exc_info=False)
            elif "symbol" in err_msg_lower: # Catch BadSymbol cases reported as InvalidOrder sometimes
                 logger.error(f"{self._colors['error']}Order weaving failed: Invalid symbol specified in parameters. {err_msg}{self._colors['reset']}", exc_info=False)
            else:
                 logger.error(f"{self._colors['error']}Order weaving failed: Flawed incantation (invalid order parameters). Review all order details. {err_msg}{self._colors['reset']}", exc_info=False)
            raise # Re-raise the specific InvalidOrder exception
        except ccxt.BadSymbol as e:
             logger.error(f"{self._colors['error']}Order weaving failed: Invalid symbol sigil '{symbol}'. Does it exist and is it active on Bybit USDT Perpetual market? {e}{self._colors['reset']}", exc_info=False)
             raise # Re-raise BadSymbol specifically
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Exchange/Network Anomaly placing order: {e}{self._colors['reset']}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected vortex placing order: {e}{self._colors['reset']}", exc_info=True)
            raise

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: Optional[int] = None) -> List[List[Union[int, float]]]:
        """Fetches OHLCV data, handling specific errors and V5 params."""
        self._check_initialized()
        # Normalize inputs
        symbol = symbol.upper()
        timeframe = timeframe.lower() # Use lowercase standard

        logger.debug(f"{self._colors['info']}# Scrying {limit} OHLCV candles for {symbol} ({timeframe})...{self._colors['reset']}")
        try:
            # Bybit V5 requires category=linear for USDT perpetuals
            params = {'category': 'linear'}
            # CCXT fetch_ohlcv timestamp is inclusive (data up to the timestamp)
            # The 'since' parameter is a start timestamp.
            # If 'since' is None, CCXT usually fetches the most recent data up to 'limit'.
            # We need 'limit' to get the desired number of candles.
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since, params=params)
            if not ohlcv:
                 # Distinguish between invalid pair and simply no data yet
                 if symbol not in self.exchange.markets:
                     logger.error(f"{self._colors['error']}Symbol '{symbol}' not found in loaded markets for OHLCV fetch.{self._colors['reset']}")
                     raise ValueError(f"Invalid symbol for OHLCV: {symbol}")
                 else:
                     logger.warning(f"{self._colors['warning']}No OHLCV data returned for {symbol} ({timeframe}). May be a new pair, inactive market, or requested range has no data.{self._colors['reset']}")
                 return [] # Return empty list if no data

            logger.debug(f"Received {len(ohlcv)} OHLCV candles.")
            # Return data as is, CCXT standard format expected: [timestamp, open, high, low, close, volume]
            return ohlcv
        except ccxt.BadSymbol as e:
            logger.error(f"{self._colors['error']}Invalid symbol sigil for OHLCV: '{symbol}'. {e}{self._colors['reset']}", exc_info=False)
            raise ValueError(f"Invalid symbol: {symbol}") from e # Re-raise as ValueError for terminal to catch
        except ccxt.ArgumentsRequired as e:
             logger.error(f"{self._colors['error']}CCXT ArgumentsRequired fetching OHLCV: {e}{self._colors['reset']}", exc_info=False)
             raise ValueError(f"Missing required arguments for OHLCV fetch: {e}") from e # Re-raise as ValueError
        except ccxt.ExchangeError as e:
            # Improve timeframe error detection by checking error message content
            err_msg = str(e).lower()
            if 'interval' in err_msg or 'candle type' in err_msg or 'invalid timeframe' in err_msg or 'period' in err_msg:
                 logger.error(f"{self._colors['error']}Invalid or unsupported time-shard '{timeframe}' for {symbol}. Check available timeframes on Bybit. {e}{self._colors['reset']}", exc_info=False)
                 # Suggest available timeframes if possible
                 tf_suggestion = ""
                 if self.exchange and self.exchange.timeframes:
                     tf_suggestion = f" Available: {list(self.exchange.timeframes.keys())}"
                 raise ValueError(f"Invalid timeframe: {timeframe}.{tf_suggestion}") from e # Re-raise as ValueError
            else:
                 logger.error(f"{self._colors['error']}CCXT Exchange Anomaly fetching OHLCV: {e}{self._colors['reset']}", exc_info=True)
                 raise # Re-raise original CCXT error
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Network Rift fetching OHLCV: {e}{self._colors['reset']}", exc_info=True)
            raise # Re-raise original CCXT error
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected vortex fetching OHLCV: {e}{self._colors['reset']}", exc_info=True)
            raise # Re-raise general error

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetches the latest ticker information for a symbol using V5 params."""
        self._check_initialized()
        symbol = symbol.upper() # Normalize
        logger.debug(f"{self._colors['info']}# Fetching current price pulse for {symbol}...{self._colors['reset']}")
        try:
            # Bybit V5 requires category=linear
            params = {'category': 'linear'}
            ticker = await self.exchange.fetch_ticker(symbol, params=params)
            logger.debug(f"Ticker data for {symbol}: {ticker}")
            return ticker
        except ccxt.BadSymbol as e:
            logger.error(f"{self._colors['error']}Invalid symbol sigil for ticker: '{symbol}'. {e}{self._colors['reset']}", exc_info=False)
            raise ValueError(f"Invalid symbol: {symbol}") from e # Re-raise as ValueError
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Exchange/Network Anomaly fetching ticker: {e}{self._colors['reset']}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected vortex fetching ticker: {e}{self._colors['reset']}", exc_info=True)
            raise

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetches open orders using V5 params, optionally filtered by symbol."""
        self._check_initialized()
        target_symbol = symbol.upper() if symbol else None # Normalize symbol if provided
        action = f"for {target_symbol}" if target_symbol else "for all symbols"
        logger.debug(f"{self._colors['info']}# Fetching active order scrolls {action}...{self._colors['reset']}")
        try:
            # Bybit V5 requires 'category': 'linear' for USDT perpetuals
            params = {'category': 'linear'}
            # CCXT fetch_open_orders usually handles the symbol filtering internally if symbol is passed
            open_orders = await self.exchange.fetch_open_orders(symbol=target_symbol, params=params)
            logger.debug(f"Found {len(open_orders)} open orders matching filter.")
            return open_orders
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Exchange/Network Anomaly fetching open orders: {e}{self._colors['reset']}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected vortex fetching open orders: {e}{self._colors['reset']}", exc_info=True)
            raise

    async def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetches open positions using V5 params, optionally filtered by symbol, and filters zero-size."""
        self._check_initialized()
        # CCXT standard fetch_positions takes a list of symbols or None for all
        symbols_list = [symbol.upper()] if symbol else None # Normalize symbol if provided
        action = f"for {symbol.upper()}" if symbol else "for all symbols"
        logger.debug(f"{self._colors['info']}# Fetching open position essences {action}...{self._colors['reset']}")
        try:
            # Bybit V5 requires 'category': 'linear'
            params = {'category': 'linear'}
            # Fetch positions for the specified symbols (or all if symbols_list is None)
            positions = await self.exchange.fetch_positions(symbols=symbols_list, params=params)
            logger.debug(f"Received {len(positions)} raw position records from exchange.")

            # Filter out zero-sized positions (CCXT standard often includes them)
            open_positions = []
            for p in positions:
                size_str = None
                pos_symbol = p.get('symbol', 'Unknown Symbol')
                try:
                    # Determine size: Check standard 'contracts', fallback 'contractSize', then Bybit V5 'info.size'
                    if p.get('contracts') is not None:
                        size_str = str(p['contracts'])
                    elif p.get('contractSize') is not None:
                         size_str = str(p['contractSize'])
                    # Check Bybit V5 'info' dict for 'size' field
                    elif 'info' in p and isinstance(p.get('info'), dict) and p['info'].get('size') is not None:
                         size_str = str(p['info']['size']) # Bybit V5 uses 'size' in info

                    # Ensure size is treated as a number for comparison, handle potential None or empty string
                    if size_str is not None and size_str != '':
                        size_float = float(size_str)
                        # Use a small tolerance for floating point comparison to zero
                        if not np.isclose(size_float, 0.0):
                            open_positions.append(p)
                        # else:
                        #    logger.debug(f"Filtered out zero-size position for {pos_symbol}")
                    # else:
                    #     logger.debug(f"Filtered out position for {pos_symbol} due to missing size info.")

                except (ValueError, TypeError) as size_err:
                    logger.warning(f"Could not parse position size '{size_str}' for symbol {pos_symbol}. Skipping position. Error: {size_err}")

            logger.info(f"Found {len(open_positions)} open (non-zero size) positions matching filter.")
            return open_positions
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            logger.error(f"{self._colors['error']}CCXT Exchange/Network Anomaly fetching positions: {e}{self._colors['reset']}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected vortex fetching positions: {e}{self._colors['reset']}", exc_info=True)
            raise


# --- Technical Analysis ---

class TechnicalAnalysis:
    """Performs technical analysis on market data."""
    def __init__(self, config: Dict):
        self.config = config
        # Ensure indicator periods are loaded correctly, use defaults if missing
        self.indicator_periods = config.get("indicator_periods", ConfigManager.DEFAULT_CONFIG["indicator_periods"]).copy()
        # Get colors from a temporary UI instance for logging/messages here
        self._colors = ConfigManager().theme_colors

        if ta is None:
            logger.warning(f"{self._colors['warning']}# pandas_ta not found. Technical Analysis Oracle is dormant.{self._colors['reset']}")
            logger.warning(f"{self._colors['warning']}Install it using: pip install pandas_ta{self._colors['reset']}")
        else:
            logger.info(f"{self._colors['primary']}# Technical Analysis Oracle initialized.{self._colors['reset']}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates technical indicators using pandas_ta based on config."""
        if ta is None:
            logger.error(f"{self._colors['error']}Cannot calculate indicators: pandas_ta library is not installed.{self._colors['reset']}")
            return df # Return original dataframe

        if df.empty:
            logger.warning(f"{self._colors['warning']}Cannot conjure indicators from empty DataFrame.{self._colors['reset']}")
            return df

        # Ensure standard column names (lowercase) and required columns exist
        df_in = df.copy() # Work on a copy
        # Convert columns to lowercase strings defensively
        df_in.columns = [str(col).lower() for col in df_in.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_in.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df_in.columns]
             logger.error(f"{self._colors['error']}DataFrame lacks required columns for TA: {missing}. Available: {list(df_in.columns)}{self._colors['reset']}")
             return df # Return original df as we cannot proceed

        # Ensure OHLCV columns are numeric, convert if possible, else drop rows with non-numeric essential data
        # Use errors='coerce' to turn non-numeric values into NaN
        for col in ['open', 'high', 'low', 'close']:
            df_in[col] = pd.to_numeric(df_in[col], errors='coerce')
        df_in['volume'] = pd.to_numeric(df_in['volume'], errors='coerce') # Volume less critical for some indicators

        # Drop rows with NaN in essential OHLC columns before calculating TA
        initial_len = len(df_in)
        df_in.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        if len(df_in) < initial_len:
             logger.warning(f"{self._colors['warning']}Dropped {initial_len - len(df_in)} rows with NaN OHLC data before TA calculation.{self._colors['reset']}")
        if df_in.empty:
             logger.error(f"{self._colors['error']}DataFrame is empty after removing NaN OHLC data. Cannot calculate TA.{self._colors['reset']}")
             return df # Return original

        logger.debug(f"{self._colors['info']}# Calculating arcane indicators on {len(df_in)} rows...{self._colors['reset']}")
        # Create a fresh copy for output to avoid modifying the input df reference unexpectedly
        df_out = df_in.copy()

        # Safely get lengths from config, providing defaults if missing or invalid type
        periods = self.indicator_periods
        try:
            # Ensure periods are integers/floats for pandas_ta, use defaults on error
            sma_short_len = int(periods.get("sma_short", ConfigManager.DEFAULT_CONFIG['indicator_periods']['sma_short']))
            sma_long_len = int(periods.get("sma_long", ConfigManager.DEFAULT_CONFIG['indicator_periods']['sma_long']))
            ema_len = int(periods.get("ema", ConfigManager.DEFAULT_CONFIG['indicator_periods']['ema']))
            rsi_len = int(periods.get("rsi", ConfigManager.DEFAULT_CONFIG['indicator_periods']['rsi']))
            bb_len = int(periods.get("bbands", ConfigManager.DEFAULT_CONFIG['indicator_periods']['bbands']))
            bb_std = float(periods.get("bbands_std", ConfigManager.DEFAULT_CONFIG['indicator_periods']['bbands_std']))
            stoch_k = int(periods.get("stoch_k", ConfigManager.DEFAULT_CONFIG['indicator_periods']['stoch_k']))
            stoch_d = int(periods.get("stoch_d", ConfigManager.DEFAULT_CONFIG['indicator_periods']['stoch_d']))
            stoch_smooth_k = int(periods.get("stoch_smooth_k", ConfigManager.DEFAULT_CONFIG['indicator_periods']['stoch_smooth_k']))
            atr_len = int(periods.get("atr", ConfigManager.DEFAULT_CONFIG['indicator_periods']['atr']))
            # MACD defaults (12, 26, 9) are standard in pandas_ta if not specified
        except (ValueError, TypeError) as e:
             logger.error(f"{self._colors['error']}Invalid indicator period type in configuration: {e}. Using defaults.{self._colors['reset']}")
             # Reset periods to defaults if conversion failed
             default_periods = ConfigManager.DEFAULT_CONFIG["indicator_periods"].copy()
             sma_short_len, sma_long_len, ema_len, rsi_len = default_periods['sma_short'], default_periods['sma_long'], default_periods['ema'], default_periods['rsi']
             bb_len, bb_std = default_periods['bbands'], default_periods['bbands_std']
             stoch_k, stoch_d, stoch_smooth_k = default_periods['stoch_k'], default_periods['stoch_d'], default_periods['stoch_smooth_k']
             atr_len = default_periods['atr']


        try:
            # Define the strategy using pandas_ta
            custom_strategy = ta.Strategy(
                name="Pyrmethus TA Set",
                description="Common indicators: SMA, EMA, RSI, MACD, BBands, Stoch, ATR",
                ta=[
                    {"kind": "sma", "length": sma_short_len},
                    {"kind": "sma", "length": sma_long_len},
                    {"kind": "ema", "length": ema_len},
                    {"kind": "rsi", "length": rsi_len},
                    {"kind": "macd"}, # Uses default lengths (12, 26, 9)
                    {"kind": "bbands", "length": bb_len, "std": bb_std},
                    {"kind": "stoch", "k": stoch_k, "d": stoch_d, "smooth_k": stoch_smooth_k},
                    {"kind": "atr", "length": atr_len},
                ]
            )
            # Apply the strategy. pandas_ta attaches methods to df.ta accessor.
            # It modifies df_out *inplace* when using the accessor.
            # Use try-except around ta.strategy in case of pandas_ta internal issues
            try:
                 df_out.ta.strategy(custom_strategy)
            except Exception as ta_strategy_err:
                 logger.error(f"{self._colors['error']}Error applying pandas_ta strategy: {ta_strategy_err}{self._colors['reset']}", exc_info=True)
                 # Return the dataframe *before* indicators were attempted
                 return df_in.copy()


            # Check if columns were actually added (strategy might fail silently or add nothing if data too short)
            added_cols = list(set(df_out.columns) - set(df_in.columns))
            if not added_cols:
                # Estimate minimum rows needed based on longest lookback
                # Ensure indicator periods are integers before using them in max()
                min_rows_needed = max(
                    sma_long_len, ema_len, rsi_len, bb_len,
                    (12 + 26 + 9), # Estimate for MACD (fast, slow, signal)
                    (stoch_k + stoch_d + stoch_smooth_k), # Estimate for Stoch
                    atr_len
                ) + 1 # +1 for calculation base

                if len(df_out) < min_rows_needed:
                     logger.warning(f"{self._colors['warning']}Pandas TA strategy ran but added no new indicator columns. Insufficient data length ({len(df_out)} rows) for configured lookback periods (longest requires ~{min_rows_needed}).{self._colors['reset']}")
                else:
                     # If enough data, maybe some other issue
                     logger.warning(f"{self._colors['warning']}Pandas TA strategy ran but added no new indicator columns despite sufficient data ({len(df_out)} rows). Check indicator configurations or library behavior.{self._colors['reset']}")
            else:
                logger.debug(f"{self._colors['success']}Successfully calculated indicators. Columns added: {added_cols}{self._colors['reset']}")
                # Round indicator values for cleaner display (only newly added columns)
                # Handle potential non-numeric columns added by TA (unlikely but possible)
                for col in added_cols:
                    if pd.api.types.is_numeric_dtype(df_out[col]):
                        # Round floating point columns only
                        if pd.api.types.is_float_dtype(df_out[col]):
                             df_out[col] = df_out[col].round(5)
                        # Integers don't need rounding

            # Reindex df_out to match the original df's index to include rows that might have been dropped initially
            # Fill potentially missing indicator values in those re-introduced rows with NaN
            # This ensures the output DataFrame has the same index as the input `df`
            df_final = df_out.reindex(df.index)
            return df_final

        except AttributeError as e:
             if "'DataFrame' object has no attribute 'ta'" in str(e) or "module 'pandas' has no attribute 'DataFrame'" in str(e):
                 logger.error(f"{self._colors['error']}Pandas TA extension not found or failed to attach. Is pandas_ta installed and imported correctly? Is pandas working?{self._colors['reset']}", exc_info=True)
             else:
                 logger.error(f"{self._colors['error']}Attribute error calculating indicators: {e}{self._colors['reset']}", exc_info=True)
             return df # Return original df on error
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected error calculating indicators: {e}{self._colors['reset']}", exc_info=True)
            return df # Return original df on error

    def calculate_pivot_points(self, df_period: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Calculates Classic Pivot Points based on the provided DataFrame.
        Expects a DataFrame with at least one row containing the High, Low, Close (HLC)
        of the *last completed* period (e.g., the previous day's HLC for daily pivots).
        """
        if df_period is None or df_period.empty:
            logger.warning(f"{self._colors['warning']}Insufficient data for pivot point calculation (DataFrame empty).{self._colors['reset']}")
            return None

        # Ensure columns are lowercase and required columns exist
        df_period_copy = df_period.copy() # Work on a copy
        df_period_copy.columns = [str(col).lower() for col in df_period_copy.columns]
        required_cols = ['high', 'low', 'close']
        if not all(col in df_period_copy.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_period_copy.columns]
            logger.error(f"{self._colors['error']}Pivot DataFrame missing required columns: {missing}. Available: {list(df_period_copy.columns)}{self._colors['reset']}")
            return None

        # Use the first row provided, assuming it's the last *completed* period's data
        if len(df_period_copy) < 1:
             logger.warning(f"{self._colors['warning']}Insufficient rows in pivot DataFrame (need at least 1 row for last completed period).{self._colors['reset']}")
             return None

        try:
            # Use the first row (index 0) passed to this function
            last_period = df_period_copy.iloc[0]
            # Use .get() and pd.to_numeric with errors='coerce' for robustness against missing or invalid data
            high = pd.to_numeric(last_period.get('high'), errors='coerce')
            low = pd.to_numeric(last_period.get('low'), errors='coerce')
            close = pd.to_numeric(last_period.get('close'), errors='coerce')

            # Check for NaN values after conversion before calculation
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                 # Attempt to get time info from index name
                 candle_time_info = last_period.name if hasattr(last_period, 'name') and pd.notna(last_period.name) else "Unknown Time"
                 # Format timestamp nicely if it is one
                 if isinstance(candle_time_info, pd.Timestamp):
                     candle_time_info = candle_time_info.strftime('%Y-%m-%d %H:%M:%S %Z') # Include timezone if available

                 logger.warning(f"{self._colors['warning']}NaN values encountered in HLC data for pivot calculation (H:{high}, L:{low}, C:{close}) for candle '{candle_time_info}'. Cannot calculate pivots.{self._colors['reset']}")
                 return None

            # Ensure values are floats for calculation (already handled by to_numeric)
            high, low, close = float(high), float(low), float(close)

            # Classic Pivot Point Calculation
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)

            # Get time info for logging/debugging
            candle_time_str = ""
            if hasattr(last_period, 'name') and pd.notna(last_period.name):
                 if isinstance(last_period.name, pd.Timestamp):
                     candle_time_str = last_period.name.strftime('%Y-%m-%d %H:%M:%S %Z')
                 else:
                     candle_time_str = str(last_period.name)
            else:
                 candle_time_str = "Unknown Time"


            logger.debug(f"{self._colors['info']}# Pivot points calculated based on H={high}, L={low}, C={close} from period ending {candle_time_str}{self._colors['reset']}")
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
            logger.error(f"{self._colors['error']}Missing HLC columns during pivot calculation access: {e}{self._colors['reset']}")
            return None
        except (ValueError, TypeError) as e:
             logger.error(f"{self._colors['error']}Data type error during pivot calculation: {e}. Ensure HLC are numeric.{self._colors['reset']}")
             return None
        except Exception as e:
            logger.error(f"{self._colors['error']}Unexpected error calculating pivot points: {e}{self._colors['reset']}", exc_info=True)
            return None

    def determine_trend(self, df: pd.DataFrame) -> str:
        """Determines a simple trend based on short vs long SMA crossover from the last two valid points."""
        # Get fresh colors instance
        colors = ConfigManager().theme_colors

        if ta is None: return f"{colors['dim']}Unknown (pandas_ta missing){colors['reset']}"
        if df.empty: return f"{colors['dim']}Unknown (No Data){colors['reset']}"

        # Get column names based on config lengths (expecting pandas_ta default naming)
        try:
            sma_short_len = int(self.indicator_periods.get('sma_short', 20))
            sma_long_len = int(self.indicator_periods.get('sma_long', 50))
        except (ValueError, TypeError):
             logger.warning(f"{colors['warning']}Invalid SMA periods in config for trend determination, using defaults 20/50.{colors['reset']}")
             sma_short_len, sma_long_len = 20, 50

        # pandas_ta names columns like 'SMA_20', 'SMA_50' (uppercase) or potentially 'sma_20' if forced lowercase
        # Check for both cases, prefer lowercase if calculate_indicators forces it.
        sma_short_col_options = [f"sma_{sma_short_len}", f"SMA_{sma_short_len}"]
        sma_long_col_options = [f"sma_{sma_long_len}", f"SMA_{sma_long_len}"]

        # Find which column names actually exist in the DataFrame
        sma_short_col = next((col for col in sma_short_col_options if col in df.columns), None)
        sma_long_col = next((col for col in sma_long_col_options if col in df.columns), None)

        if not sma_short_col or not sma_long_col:
            # Check if the expected columns are missing entirely
            missing_cols = []
            if not sma_short_col: missing_cols.append(f"SMA_{sma_short_len} (or sma_{sma_short_len})")
            if not sma_long_col: missing_cols.append(f"SMA_{sma_long_len} (or sma_{sma_long_len})")
            logger.warning(f"{colors['warning']}Required SMA columns for trend ({', '.join(missing_cols)}) not found in DataFrame. Columns available: {list(df.columns)}{colors['reset']}")
            return f"{colors['dim']}Unknown (Missing SMAs){colors['reset']}"

        # Get the last two rows where BOTH SMAs are not NaN
        # Ensure the columns are numeric before dropping NaNs (should be handled in calculate_indicators, but defensive)
        try:
             # Use pd.to_numeric with errors='coerce' and then dropna
             df[sma_short_col] = pd.to_numeric(df[sma_short_col], errors='coerce')
             df[sma_long_col] = pd.to_numeric(df[sma_long_col], errors='coerce')
             # Select columns first, then dropna
             valid_sma_df = df[[sma_short_col, sma_long_col]].dropna()
        except KeyError:
             # This should be caught above, but defensive check
             logger.error(f"{colors['error']}KeyError accessing SMA columns '{sma_short_col}' or '{sma_long_col}' during trend calculation.{colors['reset']}")
             return f"{colors['dim']}Unknown (SMA Access Error){colors['reset']}"
        except Exception as e:
             logger.error(f"{colors['error']}Error preparing SMA data for trend: {e}{colors['reset']}", exc_info=True)
             return f"{colors['dim']}Unknown (SMA Prep Error){colors['reset']}"


        if len(valid_sma_df) < 2:
             logger.debug(f"Insufficient non-NaN SMA data points ({len(valid_sma_df)}) for trend calculation.")
             return f"{colors['dim']}Unknown (Insufficient SMA Data){colors['reset']}"

        # Use iloc for positional access to the last two valid rows
        last_row = valid_sma_df.iloc[-1]
        prev_row = valid_sma_df.iloc[-2]

        trend = "Neutral / Sideways"
        color = colors['neutral'] # Default neutral color

        try:
            # Current state comparison
            # Ensure comparison is numeric, handle potential non-numeric values from .loc if coerce failed earlier
            last_short = pd.to_numeric(last_row.get(sma_short_col), errors='coerce')
            last_long = pd.to_numeric(last_row.get(sma_long_col), errors='coerce')
            prev_short = pd.to_numeric(prev_row.get(sma_short_col), errors='coerce')
            prev_long = pd.to_numeric(prev_row.get(sma_long_col), errors='coerce')

            # Check if any of the critical values are NaN after coerce
            if pd.isna(last_short) or pd.isna(last_long) or pd.isna(prev_short) or pd.isna(prev_long):
                 logger.warning(f"{colors['warning']}NaN values found in last two valid SMA data points during trend comparison. Cannot determine trend accurately.{colors['reset']}")
                 return f"{colors['dim']}Unknown (NaN SMAs){colors['reset']}"

            is_bullish_now = last_short > last_long
            is_bearish_now = last_short < last_long

            # Previous state comparison
            was_bullish_prev = prev_short > prev_long
            # Need was_bearish_prev for accurate crossover detection
            was_bearish_prev = prev_short < prev_long


            if is_bullish_now:
                trend = "Up"
                color = colors['positive']
                # Check for crossover from non-bullish (bearish or equal) in the previous valid period
                if not was_bullish_prev: # Could have been bearish or equal
                    trend += " (Bullish Crossover)"
                    color = colors['success'] # Use brighter success color for crossover
            elif is_bearish_now:
                trend = "Down"
                color = colors['negative']
                # Check for crossover from non-bearish (bullish or equal) in the previous valid period
                if not was_bearish_prev: # Could have been bullish or equal
                    trend += " (Bearish Crossover)"
                    color = colors['error'] # Use brighter error color for crossover

            logger.debug(f"{self._colors['info']}# Trend determined: {trend}{self._colors['reset']}")
            # Ensure reset code is applied at the end
            return f"{color}{trend}{colors['reset']}"

        except TypeError as e:
             logger.error(f"{colors['error']}TypeError during SMA comparison for trend: {e}. Check SMA column data types. Last values: {last_row.to_dict()}, Prev values: {prev_row.to_dict()}{colors['reset']}", exc_info=True)
             return f"{colors['dim']}Unknown (SMA Type Error){colors['reset']}"
        except Exception as e:
             logger.error(f"{colors['error']}Unexpected error determining trend: {e}{colors['reset']}", exc_info=True)
             return f"{colors['dim']}Unknown (Trend Calc Error){colors['reset']}"


    def generate_ta_summary(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generates a concise summary of key TA indicators from the last valid row."""
        summary = {}
        # Get fresh colors instance
        colors = ConfigManager().theme_colors

        if ta is None: return {"Status": f"{colors['warning']}pandas_ta missing{colors['reset']}"}
        if df.empty: return {"Status": f"{colors['warning']}No data for summary{colors['reset']}"}

        # Find the index of the last row with a valid 'close' value
        last_valid_row_idx = df['close'].last_valid_index()
        if last_valid_row_idx is None:
            logger.warning(f"{colors['warning']}No valid 'close' data found in DataFrame for TA summary.{colors['reset']}")
            return {"Status": f"{colors['warning']}No valid data points{colors['reset']}"}

        try:
             # Get the last valid row using .loc
             last = df.loc[last_valid_row_idx]
        except KeyError:
             logger.error(f"{colors['error']}Could not locate last valid row index '{last_valid_row_idx}' in DataFrame.{colors['reset']}")
             return {"Status": f"{colors['error']}Error accessing data{colors['reset']}"}


        # Standardize column access (assume lowercase from calculate_indicators or check both cases)
        df_cols = df.columns.tolist()

        # Helper to safely get value from the last valid row, checking potential column names
        def get_val(col_name_options: List[str], default="N/A") -> Any:
            for col_name in col_name_options:
                # Check if column exists AND if the value in the 'last' series for that column is not NaN
                if col_name in df_cols and pd.notna(last.get(col_name)):
                    return last.get(col_name) # Access using .get() for safety
            return default

        # Price (use lowercase 'close')
        price = get_val(['close'])
        summary['Last Price'] = f"{colors['accent']}{price:.4f}{colors['reset']}" if isinstance(price, (float, int, np.number)) else str(price) # Ensure string

        # Trend (pass the full df for history check)
        summary['Trend (SMA)'] = self.determine_trend(df) # determine_trend already returns colored string

        # RSI (e.g., RSI_14 or rsi_14)
        try: rsi_len = int(self.indicator_periods.get('rsi', 14))
        except (ValueError, TypeError): rsi_len = 14
        rsi_cols = [f"rsi_{rsi_len}", f"RSI_{rsi_len}"]
        rsi = get_val(rsi_cols)
        if isinstance(rsi, (float, int, np.number)):
            rsi_val = f"{rsi:.2f}"
            if rsi > 70: summary['RSI'] = f"{colors['negative']}{rsi_val} (Overbought){colors['reset']}"
            elif rsi < 30: summary['RSI'] = f"{colors['positive']}{rsi_val} (Oversold){colors['reset']}"
            else: summary['RSI'] = f"{colors['neutral']}{rsi_val} (Neutral){colors['reset']}"
        else: summary['RSI'] = str(rsi) # Ensure string representation

        # MACD (pandas_ta defaults: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9 or lowercase)
        hist = get_val(['macdh_12_26_9', 'MACDh_12_26_9']) # Histogram is often most useful for divergence/momentum
        signal = get_val(['macds_12_26_9', 'MACDs_12_26_9']) # Signal line
        macd_line = get_val(['macd_12_26_9', 'MACD_12_26_9']) # MACD line

        macd_str = "N/A"
        if isinstance(hist, (float, int, np.number)):
            hist_val = f"{hist:.4f}"
            if hist > 0: macd_str = f"{colors['positive']}Hist: {hist_val} (Bullish){colors['reset']}"
            elif hist < 0: macd_str = f"{colors['negative']}Hist: {hist_val} (Bearish){colors['reset']}"
            else: macd_str = f"{colors['neutral']}Hist: {hist_val} (Neutral){colors['reset']}"
            # Optionally add MACD line vs Signal line info
            if isinstance(macd_line, (float, int, np.number)) and isinstance(signal, (float, int, np.number)):
                 if macd_line > signal: macd_str += f" {colors['positive']}(M>S){colors['reset']}"
                 elif macd_line < signal: macd_str += f" {colors['negative']}(M<S){colors['reset']}"
        summary['MACD'] = macd_str


        # Stochastic (e.g., STOCHk_14_3_3, STOCHd_14_3_3 or lowercase)
        try:
            stoch_k_len = int(self.indicator_periods.get('stoch_k', 14))
            stoch_d_len = int(self.indicator_periods.get('stoch_d', 3))
            stoch_smooth_k_len = int(self.indicator_periods.get('stoch_smooth_k', 3))
        except (ValueError, TypeError):
             stoch_k_len, stoch_d_len, stoch_smooth_k_len = 14, 3, 3

        stoch_k_cols = [f"stochk_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}", f"STOCHk_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}"]
        stoch_d_cols = [f"stochd_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}", f"STOCHd_{stoch_k_len}_{stoch_d_len}_{stoch_smooth_k_len}"]
        k = get_val(stoch_k_cols)
        d = get_val(stoch_d_cols)
        stoch_str = "N/A"
        if isinstance(k, (float, int, np.number)) and isinstance(d, (float, int, np.number)):
            stoch_val = f"K:{k:.2f}, D:{d:.2f}"
            if k > 80 and d > 80: stoch_str = f"{colors['negative']}{stoch_val} (Overbought){colors['reset']}"
            elif k < 20 and d < 20: stoch_str = f"{colors['positive']}{stoch_val} (Oversold){colors['reset']}"
            # Check for crossover
            elif k > d: stoch_str = f"{colors['neutral']}{stoch_val} {colors['positive']}(K>D){colors['reset']}"
            elif k < d: stoch_str = f"{colors['neutral']}{stoch_val} {colors['negative']}(K<D){colors['reset']}"
            else: stoch_str = f"{colors['neutral']}{stoch_val}{colors['reset']}"
        summary['Stochastic'] = stoch_str

        # ATR (Volatility - e.g., ATR_14 or atr_14)
        try: atr_len = int(self.indicator_periods.get('atr', 14))
        except(ValueError, TypeError): atr_len = 14
        atr_cols = [f"atr_{atr_len}", f"ATR_{atr_len}"]
        atr = get_val(atr_cols)
        if isinstance(atr, (float, int, np.number)):
            summary['ATR'] = f"{colors['secondary']}{atr:.4f}{colors['reset']}" # Use secondary color for volatility
        else: summary['ATR'] = str(atr)

        return summary

# --- Terminal UI ---

class TerminalUI:
    """Handles the terminal user interface, menus, and input."""
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        # Colors are set dynamically based on config_manager's theme
        self.colors = config_manager.theme_colors

    def clear_screen(self):
        """Clears the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_header(self, title: str):
        """Displays a standardized header, recalculating width each time."""
        self.clear_screen()
        try:
            # Get terminal width dynamically
            term_width = os.get_terminal_size().columns
        except OSError:
            term_width = 80 # Default width if detection fails
        # Ensure minimum width for header formatting
        term_width = max(term_width, len(title) + 4) # At least title length + padding

        border = f"{self.colors['primary']}{'=' * term_width}{self.colors['reset']}"
        print(border)
        # Center title within the available width
        # Remove ANSI codes for centering calculation
        title_plain = re.sub(r'\x1b\[.*?m', '', title)
        padding = term_width - len(title_plain)
        left_padding = padding // 2
        right_padding = padding - left_padding # Handle odd padding

        # Print the title with padding, preserving embedded colors
        print(f"{' ' * left_padding}{self.colors['title']}{title}{self.colors['reset']}{' ' * right_padding}")

        print(border + "\n") # Add a newline after the bottom border


    def display_menu(self, title: str, options: List[str], prompt: str = "Choose your action") -> str:
        """
        Displays a menu and gets user choice. Handles EOFError/KeyboardInterrupt.
        This is a BLOCKING function, intended to be run in an executor thread.
        """
        # Use the dynamic header which clears the screen
        self.display_header(title)
        for i, option in enumerate(options, 1):
            # Option text might already contain color codes (e.g., for disabled items or values)
            # Ensure reset is applied correctly after potentially colored option text
            print(f"{self.colors['menu_highlight']}{i}. {self.colors['menu_option']}{option}{self.colors['reset']}")

        while True:
            try:
                choice = input(f"\n{self.colors['input_prompt']}{prompt} (1-{len(options)}): {self.colors['reset']}").strip()
                if choice.isdigit():
                    choice_int = int(choice)
                    if 1 <= choice_int <= len(options):
                        logger.debug(f"Menu '{title}' choice: {choice}")
                        return choice
                    else:
                         # Use print_warning for invalid input feedback (uses correct theme colors)
                         # Need to print directly as this is a blocking function
                         print(f"{self.colors['warning']}Warning: Invalid input. Please enter a number between 1 and {len(options)}.{self.colors['reset']}")
                else:
                     print(f"{self.colors['warning']}Warning: Invalid input. Please enter a number.{self.colors['reset']}")

            except EOFError:
                 logger.warning("EOF detected in display_menu, signalling exit request.")
                 # Raise EOFError to be caught by the async executor wrapper
                 raise EOFError("EOF detected during menu input.")
            except KeyboardInterrupt:
                 logger.warning("KeyboardInterrupt caught during menu input, signalling exit request.")
                 # Raise KeyboardInterrupt to be caught by the async executor wrapper
                 raise KeyboardInterrupt("KeyboardInterrupt during menu input.")
            except Exception as e:
                 # Catch any unexpected error during input itself
                 logger.error(f"Unexpected error during menu input: {e}", exc_info=True)
                 print(f"{self.colors['error']}An unexpected error occurred during input: {e}{self.colors['reset']}")
                 # Continue loop to try again


    def get_input(self, prompt: str, default: Optional[str] = None, required: bool = True, input_type: type = str, validation_func: Optional[Callable[[Any], Union[bool, str]]] = None) -> Any:
        """
        Gets validated user input. Handles EOFError/KeyboardInterrupt, type conversion, and validation.
        This is a BLOCKING function, intended to be run in an executor thread.
        Returns the validated, type-converted value, or None if not required and input is blank.
        Raises EOFError or KeyboardInterrupt if encountered.
        Raises ValueError or TypeError on conversion issues.
        """
        while True:
            prompt_full = f"{self.colors['input_prompt']}{prompt}"
            if default is not None:
                # Use accent color for the default value hint
                prompt_full += f" [{self.colors['accent']}default: {default}{self.colors['input_prompt']}]"
            prompt_full += f": {self.colors['reset']}" # Reset color before user input starts

            try:
                user_input = input(prompt_full).strip()
            except EOFError:
                 logger.warning(f"EOF detected getting input for '{prompt}'.")
                 raise EOFError("Input stream closed unexpectedly.") # Propagate EOF
            except KeyboardInterrupt:
                 logger.warning(f"KeyboardInterrupt caught getting input for '{prompt}'.")
                 raise KeyboardInterrupt # Propagate interrupt
            except Exception as e:
                 # Catch any unexpected error during input itself
                 logger.error(f"Unexpected error during input for '{prompt}': {e}", exc_info=True)
                 print(f"{self.colors['error']}An unexpected error occurred during input: {e}{self.colors['reset']}")
                 continue # Continue loop to try again

            # Handle default value usage
            if not user_input and default is not None:
                user_input = str(default) # Use default value as string initially
                logger.debug(f"Input for '{prompt}' using default: {default}")

            # Handle required input that is left blank *after* checking for default
            if required and not user_input:
                # Use print_error directly here as it's synchronous
                print(f"{self.colors['error']}Error: Input is required.{self.colors['reset']}")
                continue # Ask again

            # Handle optional input that is left blank *after* checking for default
            if not user_input and not required:
                logger.debug(f"Optional input for '{prompt}' left blank.")
                return None # Return None for blank optional input

            # Proceed with conversion and validation (even if default was used)
            value: Any = None # Initialize value
            try:
                if input_type == bool:
                    # More robust boolean conversion
                    if user_input.lower() in ('true', '1', 't', 'y', 'yes'): value = True
                    elif user_input.lower() in ('false', '0', 'f', 'n', 'no'): value = False
                    else: raise ValueError("Invalid boolean value. Use True/False, Yes/No, 1/0.")
                elif input_type == str:
                    value = user_input # Already a string
                else:
                    # Attempt conversion to the target type (e.g., float, int)
                    value = input_type(user_input)

                # Perform validation if a function is provided
                if validation_func:
                     # Pass the *converted* value to the validation function
                     validation_result = validation_func(value)
                     if isinstance(validation_result, str): # Validation func returned an error message string
                         print(f"{self.colors['error']}Error: {validation_result}{self.colors['reset']}")
                         continue # Ask again
                     elif validation_result is False: # Validation func returned generic False
                         print(f"{self.colors['error']}Error: Input validation failed.{self.colors['reset']}")
                         continue # Ask again
                     # If validation_result is True or None (implicit success), proceed

                # If conversion and validation passed
                logger.debug(f"Input for '{prompt}': {repr(value)} (type: {input_type})")
                return value

            except (ValueError, TypeError) as e:
                # Provide more specific feedback for common conversion errors
                if input_type == float or input_type == int:
                    print(f"{self.colors['error']}Error: Invalid input. Expected a number ({input_type.__name__}). Details: {e}{self.colors['reset']}")
                else:
                    print(f"{self.colors['error']}Error: Invalid input format. Expected {input_type.__name__}. Details: {e}{self.colors['reset']}")
                # Loop continues to ask again
            except Exception as e: # Catch errors during validation_func execution specifically
                print(f"{self.colors['error']}Error during input validation step: {e}{self.colors['reset']}")
                logger.error(f"Exception in validation_func for prompt '{prompt}': {e}", exc_info=True)
                # Loop continues to ask again

    # --- Direct Print Helpers (Synchronous) ---
    # These can be called directly from blocking functions like get_input or display_menu
    # or from synchronous parts of the code.

    def print_error(self, message: str):
        """Prints an error message directly (synchronous)."""
        print(f"{self.colors['error']}Error: {message}{self.colors['reset']}")
        # Log the core message without the "Error:" prefix if logger adds levelname
        logger.error(message)

    def print_success(self, message: str):
        """Prints a success message directly (synchronous)."""
        print(f"{self.colors['success']}Success: {message}{self.colors['reset']}")
        logger.info(message) # Log success messages too

    def print_warning(self, message: str):
        """Prints a warning message directly (synchronous)."""
        print(f"{self.colors['warning']}Warning: {message}{self.colors['reset']}")
        logger.warning(message)

    def print_info(self, message: str):
        """Prints an informational message directly (synchronous)."""
        print(f"{self.colors['info']}{message}{self.colors['reset']}")
        # Optionally log info messages if needed for detailed flow tracking
        # logger.info(message)

    # --- Table Printing ---

    def _contains_ansi_codes(self, text: Any) -> bool:
        """Checks if a given value, when converted to string, contains ANSI escape codes."""
        if not isinstance(text, str):
             try:
                 text = str(text)
             except Exception: # Catch errors during string conversion
                 return False # Cannot convert to string, assume no ANSI codes
        # Check for the ANSI escape sequence introducer (\x1b) followed by '['
        return '\x1b' in text and '[' in text

    def print_table(self, data: Union[pd.DataFrame, List[Dict], Dict], title: Optional[str] = None, float_format: str = '{:.4f}', index: bool = False):
        """Prints data in a formatted table using tabulate if available, with improved handling for colored strings."""
        if title:
            # Print title using table_header color
            print(f"\n{self.colors['table_header']}{'--- ' + title + ' ---'}{self.colors['reset']}")

        if data is None or \
           (isinstance(data, (list, dict)) and not data) or \
           (isinstance(data, pd.DataFrame) and data.empty):
             self.print_warning("No data to display.")
             return

        headers: Union[str, List[str]] = "firstrow" # Default for tabulate if needed
        table_data: Optional[List[List[Any]]] = None
        contains_color = False # Flag to detect colored strings

        try:
            # --- Prepare Data and Detect Color ---
            if isinstance(data, pd.DataFrame):
                df = data # Keep reference to original df
                # Check entire DataFrame for ANSI codes more efficiently
                # Apply the check element-wise. .any().any() checks if any cell contains color.
                if df.applymap(self._contains_ansi_codes).any().any():
                     contains_color = True
                     logger.debug("Detected ANSI color codes in DataFrame.")

                # Prepare headers and data for tabulate or fallback
                headers = list(df.columns)
                if index:
                    # Add index as a column for printing
                    df_to_print = df.reset_index()
                    # Ensure index name(s) are strings
                    index_names = [str(name) for name in df_to_print.columns[:df.index.nlevels]]
                    headers = index_names + headers[df.index.nlevels:] # Correct headers after adding index
                    table_data = df_to_print.values.tolist()
                else:
                    df_to_print = df
                    table_data = df_to_print.values.tolist()


            elif isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
                # Convert list of dicts
                if not data: # Should be caught above, but double check
                    self.print_warning("No data to display.")
                    return
                # Use keys from the first dict as headers, assuming consistency
                headers = list(data[0].keys())
                table_data = []
                for item in data:
                     row = []
                     for h in headers:
                         value = item.get(h)
                         if self._contains_ansi_codes(value):
                             contains_color = True
                         row.append(value)
                     table_data.append(row)
                if contains_color: logger.debug("Detected ANSI color codes in list of dicts.")

            elif isinstance(data, dict) and data:
                 # Format dictionary as a two-column table (Parameter, Value)
                 headers = ["Parameter", "Value"]
                 table_data = []
                 for key, value in data.items():
                      if self._contains_ansi_codes(value):
                           contains_color = True
                      # Append key (as string) and value (original type)
                      table_data.append([str(key), value])
                 if contains_color: logger.debug("Detected ANSI color codes in dictionary values.")

            else:
                # Handle unsupported types gracefully
                self.print_error(f"Unsupported data type for print_table: {type(data)}")
                print(repr(data)) # Show representation for debugging
                return

            # --- Choose Printing Method ---

            # Use Tabulate if available AND no color codes detected
            if tabulate and table_data is not None and not contains_color:
                # Define float formatting function specifically for tabulate
                # Tabulate's numalign='right' and stralign='left' work well
                def fmt_val_tabulate(val):
                    if isinstance(val, (float, np.floating)):
                        try: return float_format.format(val)
                        except (ValueError, TypeError): return str(val) # Fallback
                    # Handle None explicitly for tabulate's missingval
                    return val if val is not None else None # Pass None to tabulate

                # Apply formatting to the data *before* passing to tabulate
                # Use a list comprehension for efficiency
                formatted_table_data = [[fmt_val_tabulate(item) for item in row] for row in table_data]

                # Use a visually appealing format
                table_fmt = "fancy_grid" # Or "pretty", "simple", "github", etc.
                # Use neutral color for the table body
                print(f"{self.colors['neutral']}{tabulate(formatted_table_data, headers=headers, tablefmt=table_fmt, numalign='right', stralign='left', missingval='N/A')}{self.colors['reset']}")

            # Use Pandas built-in string conversion if data is DataFrame (handles color better than tabulate)
            elif isinstance(data, pd.DataFrame):
                 if not tabulate: self.print_info(f"{self.colors['dim']}(Install 'tabulate' library for potentially better table format: pip install tabulate){self.colors['reset']}")
                 if contains_color: self.print_info(f"{self.colors['dim']}(Using pandas print due to colored content){self.colors['reset']}")
                 try: term_width = os.get_terminal_size().columns
                 except OSError: term_width = 120 # Wider default fallback width
                 # Configure pandas display options for better CLI output
                 with pd.option_context('display.max_rows', None, # Show all rows
                                       'display.max_columns', None, # Show all columns
                                       'display.width', term_width, # Use detected width
                                       'display.max_colwidth', None, # Prevent column wrapping
                                       'display.expand_frame_repr', True, # Prevent line wrapping if possible
                                       'display.colheader_justify', 'left', # Left-align headers
                                       # Apply float format ONLY if no colors are present (to avoid double formatting)
                                       'display.float_format', (lambda x: float_format.format(x)) if float_format and not contains_color else None
                                       ):
                     # Use to_string for better control than direct print(df)
                     # Pass the DataFrame adjusted for index/no-index printing
                     print(f"{self.colors['neutral']}{df_to_print.to_string(index=index, na_rep='N/A')}{self.colors['reset']}")

            # Fallback for list/dict if tabulate missing OR data contains color
            elif table_data is not None:
                 if not tabulate: self.print_info(f"{self.colors['dim']}(Install 'tabulate' library for potentially better table format: pip install tabulate){self.colors['reset']}")
                 if contains_color: self.print_info(f"{self.colors['dim']}(Using basic print due to colored content){self.colors['reset']}")

                 # Basic manual print (less aligned but preserves color)
                 # Determine column widths (roughly) for basic alignment
                 # Need to calculate width based on plain text length to align correctly
                 col_widths = [len(re.sub(r'\x1b\[.*?m', '', str(h))) for h in headers]
                 for row in table_data:
                     for i, item in enumerate(row):
                         # Need string length without ANSI codes for width calculation
                         item_str = str(item if item is not None else 'N/A')
                         plain_text = re.sub(r'\x1b\[.*?m', '', item_str)
                         col_widths[i] = max(col_widths[i], len(plain_text))

                 # Print header row
                 header_line = " | ".join(f"{str(h):<{col_widths[i]}}" for i, h in enumerate(headers))
                 print(f"{self.colors['table_header']}{header_line}{self.colors['reset']}")
                 # Print separator line matching header width
                 print("-" * (sum(col_widths) + 3 * (len(headers) - 1))) # 3 spaces per separator (' | ')

                 # Print data rows
                 for row in table_data:
                     row_items = []
                     for i, item in enumerate(row):
                         item_str = str(item if item is not None else 'N/A') # Default string representation, handle None
                         # Apply float formatting only if it's a number AND no color codes present
                         if not self._contains_ansi_codes(item) and isinstance(item, (float, np.floating)):
                             try: item_str = float_format.format(item)
                             except (ValueError, TypeError): pass # Keep original string if format fails

                         # Calculate padding based on plain text length
                         plain_text = re.sub(r'\x1b\[.*?m', '', item_str)
                         padding = " " * (col_widths[i] - len(plain_text))
                         # Append the original (potentially colored) string and padding
                         row_items.append(item_str + padding)

                     print(f"{self.colors['neutral']}{' | '.join(row_items)}{self.colors['reset']}")

        except Exception as e:
            logger.error(f"{self.colors['error']}Error during print_table execution: {e}{self.colors['reset']}", exc_info=True)
            self.print_error(f"An error occurred while trying to display the table: {e}")
            print("--- Raw Data ---")
            print(repr(data)) # Print raw data representation for debugging
            print("----------------")


    def wait_for_enter(self, prompt: str = "Press Enter to continue..."):
        """
        Pauses execution until Enter is pressed. Handles EOFError/KeyboardInterrupt.
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
        except Exception as e:
             # Catch any unexpected error during input itself
             logger.error(f"Unexpected error during wait for enter: {e}", exc_info=True)
             print(f"{self.colors['error']}An unexpected error occurred: {e}{self.colors['reset']}")
             # Don't re-raise, just let the function return


    def display_chart(self, data: List[float], title: str):
        """Displays a simple ASCII chart using asciichartpy with theme color."""
        if not data:
            self.print_warning(f"No data available for chart: {title}")
            return

        chart_height = self.config_manager.config.get("chart_height", 10)
        print(f"\n{self.colors['table_header']}{'--- ' + title + ' ---'}{self.colors['reset']}")
        try:
            # Clean data: ensure numeric, replace NaN/inf with previous valid value for plotting continuity
            plot_data = []
            last_valid = np.nan
            for d in data:
                # Check if it's a number type (int, float, numpy number) and finite
                if isinstance(d, (int, float, np.number)) and np.isfinite(d):
                    last_valid = float(d) # Convert to float for consistency
                    plot_data.append(last_valid)
                elif not np.isnan(last_valid): # Use last valid value if current is bad/non-numeric
                    plot_data.append(last_valid)
                # If last_valid is still NaN (e.g., at the start), append NaN.
                # asciichartpy seems to handle internal NaNs/missing points reasonably,
                # but filling helps maintain plot continuity.
                else:
                    plot_data.append(np.nan)

            # Check if any valid points remain after cleaning
            if not any(np.isfinite(d) for d in plot_data if not np.isnan(d)):
                 self.print_warning(f"No valid numeric data points for chart: {title}")
                 return

            # Get the primary color code from the theme's colorama Fore object (e.g., '\x1b[36m' for cyan)
            primary_color_ansi = self.colors['primary']
            # Extract the number code (e.g., '36') from the ANSI sequence
            primary_color_code_match = re.search(r'\x1b\[(\d+)m', primary_color_ansi)
            chart_color_code_str = primary_color_code_match.group(1) if primary_color_code_match else None

            # Map colorama number code string to asciichartpy color constant
            # asciichartpy colors are simple integers (0-7) or constants.
            # Colorama codes are 30-37 (standard) or 90-97 (bright).
            color_map = {
                '30': asciichart.black,   '90': asciichart.default, # Default often black/grey
                '31': asciichart.red,     '91': asciichart.red,
                '32': asciichart.green,   '92': asciichart.green,
                '33': asciichart.yellow,  '93': asciichart.yellow,
                '34': asciichart.blue,    '94': asciichart.blue,
                '35': asciichart.magenta, '95': asciichart.magenta,
                '36': asciichart.cyan,    '96': asciichart.cyan,
                '37': asciichart.white,   '97': asciichart.white,
            }
            # Default to asciichart.default if code not found or invalid
            chart_color = color_map.get(chart_color_code_str, asciichart.default)

            # Generate chart using asciichartpy
            # Provide configuration including height and color list (even if just one color)
            chart_config = {'height': chart_height, 'colors': [chart_color]}
            chart = asciichart.plot(plot_data, cfg=chart_config)
            # Print the chart - asciichartpy includes ANSI codes directly for the specified color
            print(chart)

        except ImportError:
             self.print_error("Failed to generate chart: 'asciichartpy' library not found.")
             logger.error("Asciichart error: Library not imported.")
        except Exception as e:
            self.print_error(f"Failed to generate chart: {e}")
            logger.error(f"Asciichart error for title '{title}': {e}", exc_info=True)


# --- Main Trading Terminal Application ---

class TradingTerminal:
    """Main class for the trading terminal application."""
    def __init__(self):
        self.config_manager = ConfigManager()
        # Pass the config_manager instance to UI so UI can access colors dynamically
        self.ui = TerminalUI(self.config_manager)
        self.credentials: Optional[APICredentials] = None
        self.exchange_client: Optional[BybitFuturesCCXTClient] = None
        # Pass config_manager config dict to TA analyzer
        self.ta_analyzer = TechnicalAnalysis(self.config_manager.config)
        self._running = False # Flag to signal the main loop to continue
        self._shutdown_event = asyncio.Event() # Event to signal shutdown requested
        self._active_tasks: set[asyncio.Task] = set() # Store tasks created by _create_task
        self._final_exit_code = 0 # Store exit code set by shutdown

    def _create_task(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Creates an asyncio task, names it (if possible), tracks it, and adds completion logging."""
        # Ensure name is provided for better debugging
        task_name = name or f"Task-{id(coro)}"
        if sys.version_info >= (3, 8):
            task = asyncio.create_task(coro, name=task_name)
        else:
            task = asyncio.create_task(coro) # name param not available < 3.8

        self._active_tasks.add(task)

        # Define callback to remove task from set and log completion/errors
        def _task_done_callback(fut: asyncio.Task):
            self._active_tasks.discard(fut) # Remove task from set

            # Use a try-except block to prevent exceptions in the callback from crashing the loop
            try:
                # Retrieve the result or exception to clear it and prevent warnings
                exc = fut.exception()
                task_name_cb = fut.get_name() if sys.version_info >= (3, 8) else f"Task-{id(fut)}"

                if isinstance(exc, asyncio.CancelledError):
                    # Cancellation is expected during shutdown, often debug level is fine
                    logger.debug(f"Task '{task_name_cb}' was cancelled.")
                elif exc:
                    # Log other exceptions at error level
                    logger.error(f"Task '{task_name_cb}' completed with error: {exc}", exc_info=exc)
                # else: # Task completed successfully
                #     logger.debug(f"Task '{task_name_cb}' completed successfully.")

            except Exception as e:
                 # Error within the callback itself - this is serious
                 logger.critical(f"Error in task done callback for '{task_name}': {e}", exc_info=True)


        task.add_done_callback(_task_done_callback)
        logger.debug(f"Task '{task_name}' created and tracked.")
        return task

    def _setup_signal_handlers(self):
        """Sets up signal handlers for graceful shutdown (SIGINT, SIGTERM)."""
        loop = asyncio.get_running_loop()
        signals_to_handle = (signal.SIGINT, signal.SIGTERM)

        def signal_handler(sig):
            """Wrapper function to schedule shutdown asynchronously."""
            # Use logger directly as UI might not be safe in signal handler context
            logger.info(f"Received signal {sig.name}. Initiating graceful shutdown...")
            # Check if shutdown is already in progress to avoid duplicate actions
            if not self._shutdown_event.is_set():
                 # Schedule the shutdown coroutine without blocking the handler.
                 # Use create_task to ensure it runs on the loop and is tracked (though shutdown cancels tasks).
                 # Use exit code 1 for signals indicating interruption
                 self._create_task(self.shutdown(signal=sig, exit_code=1), name=f"ShutdownHandler_{sig.name}")
                 # Setting _running false here signals the main loop to check _shutdown_event quickly
                 self._running = False
            else:
                 logger.debug(f"Shutdown already in progress, ignoring signal {sig.name}")

        for sig in signals_to_handle:
            try:
                # functools.partial is used to pass the signal object to the handler
                loop.add_signal_handler(sig, partial(signal_handler, sig))
                logger.debug(f"Signal handler successfully set for {sig.name}")
            except NotImplementedError:
                 # Common on Windows for SIGTERM/SIGINT in basic Python console environments
                 # Use print here as UI might not be fully initialized/safe
                 print(f"{self.ui.colors['warning']}Warning: Signal handler for {sig.name} not fully supported on this platform (e.g., Windows console). Use Exit menu or Ctrl+C/Ctrl+Break.{self.ui.colors['reset']}", file=sys.stderr)
            except ValueError as e:
                 # Might happen if loop is closing or signal already handled elsewhere
                 logger.warning(f"{self.ui.colors['warning']}Signal handler for {sig.name} could not be set (may already be handled or loop issue): {e}{self.ui.colors['reset']}")

        logger.info(f"{self.ui.colors['primary']}# Signal conduits established for graceful departure.{self.ui.colors['reset']}")


    async def setup_credentials(self) -> bool:
        """Loads API credentials from .env file, performs basic validation."""
        env_path = '.env'
        # Ensure load_dotenv uses the correct path and overrides existing env vars if needed
        load_dotenv(dotenv_path=env_path, override=True)

        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        testnet_str = os.getenv('TESTNET', 'False') # Default to 'False' if not set

        # Robust boolean conversion for TESTNET environment variable
        testnet = str(testnet_str).strip().lower() in ('true', '1', 't', 'y', 'yes')

        # Check if keys are missing or still contain obvious placeholders
        placeholders = ['your_api_key_here', 'your_api_secret_here', '', None]
        key_invalid = api_key in placeholders
        secret_invalid = api_secret in placeholders

        if key_invalid or secret_invalid:
            error_parts = []
            if key_invalid: error_parts.append("BYBIT_API_KEY")
            if secret_invalid: error_parts.append("BYBIT_API_SECRET")
            missing_vars = " and ".join(error_parts)
            logger.error(f"{missing_vars} not found or is a placeholder in '{env_path}' file.")
            # Use UI print methods here as this runs before full async loop usually
            self.ui.print_error(f"{missing_vars} not found or not configured in '{env_path}' file.")
            self.ui.print_warning(f"Please ensure {missing_vars} are correctly set in the '{env_path}' file.")
            self.ui.print_info("Obtain keys from Bybit: Account -> API Management.")
            self.ui.print_info("Ensure key has permissions for CONTRACT TRADE (and Read-Only for Balance/Positions).")
            self.credentials = None
            return False
        else:
            # Strip potential whitespace from keys/secrets before storing
            self.credentials = APICredentials(api_key.strip(), api_secret.strip(), testnet)
            logger.info(f"{self.ui.colors['success']}API credential sigils loaded successfully. Testnet mode: {testnet}{self.ui.colors['reset']}")
            return True


    async def initialize(self):
        """Initializes the terminal, including credential setup and API client connection."""
        self.ui.display_header("Pyrmethus Bybit Terminal - Initialization")
        self.ui.print_info(f"{self.ui.colors['primary']}# Awakening terminal energies...{self.ui.colors['reset']}")

        # Setup credentials first
        if not await self.setup_credentials():
            # setup_credentials already printed detailed errors
            self.ui.print_warning("Running in limited mode without valid API credentials.")
            self.ui.print_info("Authenticated actions (trading, balance, positions, orders) will be unavailable.")
            # Don't wait, proceed to menu with limited options. Client remains None.
            # Give user a moment to read before showing menu
            await asyncio.sleep(2.0)
            return # Exit initialization early

        # Credentials seem valid structurally, attempt to initialize client
        # Check defensively if credentials object was actually set
        if not self.credentials:
            logger.error("Internal Error: Credentials object is None after successful setup check.")
            self.ui.print_error("Internal error: Credentials not set despite passing initial checks.")
            # Trigger shutdown on critical internal inconsistency
            # Use create_task as initialize is an async function itself
            self._create_task(self.shutdown(exit_code=1, signal="InitCredsInconsistency"), name="ShutdownOnInitCredsFail")
            return # Exit initialization early

        # Create the client instance, passing the loaded config
        self.exchange_client = BybitFuturesCCXTClient(self.credentials, self.config_manager.config)
        try:
            # Initialize the client (connects, loads markets, handles testnet)
            await self.exchange_client.initialize()
            # Check if initialization succeeded internally (sets _initialized flag)
            if not self.exchange_client._initialized:
                 # This case might occur if initialize() caught an error but didn't re-raise ConnectionError
                 # Or if market loading failed silently after retries
                 raise ConnectionError("Client initialization sequence failed internally (check logs for details).")

            self.ui.print_success("Exchange connection established and markets loaded.")
            # Optionally show brief status before main menu
            await asyncio.sleep(1.0)
            # Proceed to menu

        except ConnectionError as e:
            # Specific connection/auth/market load errors handled during initialize() should raise ConnectionError
            self.ui.print_error(f"Failed to initialize exchange client: {e}")
            self.ui.print_warning("Proceeding without a fully functional exchange client.")
            self.ui.print_info("Authenticated features will likely fail. Check logs for details.")
            # Ensure client is properly cleaned up if init failed partially
            if self.exchange_client:
                 await self.exchange_client.close() # Explicitly close and reset state
            self.exchange_client = None # Set client to None to reflect failed state
            # Give user time to read before showing limited menu
            await asyncio.sleep(3.0)
            # Proceed to menu with limited options

        except Exception as e:
            # Catch any other unexpected error during the initialization process itself
            self.ui.print_error(f"Unexpected critical error during initialization: {e}")
            logger.critical("Critical initialization error", exc_info=True)
            # Trigger immediate shutdown if initialization fails critically
            # Use create_task to avoid awaiting shutdown within the exception handler
            self._create_task(self.shutdown(exit_code=1, signal="InitCriticalFailure"), name="ShutdownOnInitCriticalFail")
            # Ensure client is cleaned up if object exists
            if self.exchange_client: await self.exchange_client.close()
            self.exchange_client = None


    async def shutdown(self, signal=None, exit_code=0):
        """Gracefully shuts down the application, cancelling tasks and closing connections."""
        # Prevent concurrent shutdown calls using the event flag
        if self._shutdown_event.is_set():
             signal_name = f"signal {getattr(signal, 'name', signal)}" if signal else "request"
             logger.debug(f"Shutdown already in progress, ignoring duplicate trigger from {signal_name}")
             return

        # --- Start Shutdown Sequence ---
        self._running = False # Signal loops to stop FIRST
        self._shutdown_event.set() # Signal waiters SECOND
        self._final_exit_code = exit_code # Store intended exit code

        signal_name = f"signal {getattr(signal, 'name', signal)}" if signal else "menu/error request"
        logger.info(f"Shutdown sequence initiated by {signal_name}... Setting final exit code to {exit_code}")
        # Print shutdown message using UI (if possible/safe)
        # Use a direct print as UI object might be involved in tasks being cancelled
        # Use get() with defaults in case config_manager or ui is somehow partially initialized/cleaned up
        colors = getattr(getattr(self, 'ui', None), 'colors', ConfigManager.DEFAULT_CONFIG['theme_colors'])
        print(f"\n{colors.get('primary', Fore.CYAN)}# Banishing terminal... Please wait.{colors.get('reset', Style.RESET_ALL)}")


        # --- Cancel Active Background Tasks ---
        # Get the currently running task (the shutdown task itself) to avoid cancelling it
        current_task = asyncio.current_task()
        # Make a copy of the active tasks set, as it might be modified during iteration by callbacks
        # Only cancel tasks that are NOT the current task AND are NOT already done
        tasks_to_cancel = {task for task in self._active_tasks if task is not current_task and not task.done()}

        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} active background tasks...")
            # asyncio.gather can await cancellation, use return_exceptions=True
            # Note: tasks might still be in the set until their done_callback runs *after* gather finishes
            results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

            cancelled_count = 0
            error_count = 0
            completed_normally_count = 0

            # Process results to log status of each task
            # results corresponds to the order tasks were passed to gather
            for i, result in enumerate(results):
                 # Need to map result back to the original task. Iterating the original set copy works.
                 # Or just use a counter/index if the order is guaranteed (gather does preserve order).
                 # Accessing task details might be tricky if the task object state changes immediately after completion/cancellation.
                 # Logging within the done_callback is more reliable for task-specific outcomes.
                 if isinstance(result, asyncio.CancelledError):
                     cancelled_count += 1
                     # Task's done_callback will log the cancellation
                 elif isinstance(result, Exception):
                     error_count += 1
                     # Task's done_callback will log the error
                 else:
                     completed_normally_count += 1
                     # Task's done_callback will log successful completion if configured

            logger.info(f"Background tasks processing complete: {cancelled_count} cancelled, {completed_normally_count} completed normally, {error_count} errors logged by callbacks.")
        else:
            logger.info("No active background tasks needed cancellation.")

        # --- Close Exchange Client Connection ---
        # Check if the exchange client object exists before trying to close it
        if self.exchange_client:
            logger.info("Closing exchange client connection...")
            # Call the client's close method, which handles state and is idempotent
            await self.exchange_client.close()
        else:
            logger.debug("Exchange client was not initialized or already closed.")


        logger.info(f"Terminal shutdown sequence complete. Final exit code set to: {self._final_exit_code}")
        # The final "Terminal has faded" message is printed outside this coroutine in the main runner


    async def run(self):
        """Runs the main terminal loop, handling menus and actions."""
        # Setup signal handlers early in the run process
        self._setup_signal_handlers()

        # Perform initialization (credentials, client connection)
        await self.initialize()

        # Check if shutdown was triggered during initialization (e.g., critical error, failed connection)
        if self._shutdown_event.is_set():
             logger.warning("Shutdown triggered during initialization phase, exiting run loop.")
             return # Don't start the main menu loop

        # Set running flag only after successful initialization or proceeding in limited mode
        self._running = True
        logger.info("Starting main terminal loop.")

        while self._running:
            # Check shutdown flag at the start of each loop iteration
            # This check is crucial for breaking out of the loop quickly after shutdown is signalled
            if self._shutdown_event.is_set():
                logger.info("Shutdown detected at start of loop, breaking.")
                break # Exit the loop cleanly

            menu_task = None
            shutdown_wait_task = None
            try:
                # --- Wait for Menu Choice or Shutdown Signal ---
                # Create task to display menu and get input (runs UI in executor)
                # This task will block a worker thread but not the main event loop
                menu_task = self._create_task(self.ui_display_menu_async(), name="MainMenuDisplay")
                # Create a separate task that simply waits for the shutdown event to be set
                shutdown_wait_task = self._create_task(self._shutdown_event.wait(), name="ShutdownWait")

                # Use asyncio.wait to efficiently wait for *either* task to complete
                # return_when=asyncio.FIRST_COMPLETED means we get results as soon as one task finishes
                done, pending = await asyncio.wait(
                    {menu_task, shutdown_wait_task},
                    return_when=asyncio.FIRST_COMPLETED
                )

                choice: Optional[str] = None
                shutdown_triggered_during_wait = False

                # --- Process Completion Results ---
                # Check if the shutdown wait task completed (meaning shutdown was signalled)
                if shutdown_wait_task in done:
                     shutdown_triggered_during_wait = True
                     logger.info("Shutdown event triggered completion while waiting for menu.")
                     # If the menu task was still running (pending), it needs to be cancelled
                     if menu_task in pending:
                         logger.debug("Cancelling pending menu task due to shutdown signal.")
                         menu_task.cancel()
                         # Await the cancelled task briefly to ensure it cleans up and exception is retrieved
                         try: await asyncio.wait_for(menu_task, timeout=0.5)
                         except (asyncio.TimeoutError, asyncio.CancelledError): pass # Expected outcomes

                # Check if the menu task completed (user made a choice or an error occurred)
                if menu_task in done:
                    try:
                        # Retrieve the result (user's choice string) or raise exception if task failed
                        choice = await menu_task # This will raise if the task was cancelled or raised an exception
                        logger.debug(f"Menu task completed, choice: {choice}")
                    except asyncio.CancelledError:
                         # This means the menu task was cancelled, likely by the shutdown logic above
                         logger.info("Menu display task was cancelled.")
                         # If shutdown wasn't the trigger, this is unexpected. Log it.
                         if not shutdown_triggered_during_wait:
                             logger.warning("Menu task cancelled unexpectedly (not by shutdown signal).")
                         # Ensure loop breaks if menu cancelled for any reason (e.g., signal)
                         shutdown_triggered_during_wait = True
                    except (EOFError, KeyboardInterrupt) as e:
                         # These are raised by the UI methods via the async executor wrapper if user signals exit
                         logger.info(f"'{type(e).__name__}' received from menu input, initiating shutdown.")
                         # Trigger shutdown if it's not already in progress
                         if not self._shutdown_event.is_set(): # Avoid duplicate shutdown calls
                             # Use exit code 1 for interruption
                             self._create_task(self.shutdown(signal=type(e).__name__, exit_code=1), name=f"ShutdownOnMenu_{type(e).__name__}")
                         shutdown_triggered_during_wait = True # Ensure loop breaks immediately
                    except Exception as e:
                         # Catch any other unexpected errors from the menu display/input logic itself
                         logger.error(f"Error retrieving menu choice: {e}", exc_info=True)
                         self.ui.print_error(f"An error occurred in the menu display/input: {e}")
                         # Pause briefly before retrying the menu, using the async helper
                         await self.ui_wait_for_enter_async()
                         continue # Go to next loop iteration to retry displaying the menu

                # Cancel any remaining pending tasks (e.g., if menu finished first, cancel the shutdown wait task)
                # This cleans up the other task that didn't finish first in asyncio.wait
                for task in pending:
                    if not task.done(): # Check again if it finished while we were processing the first completed task
                         task.cancel()
                         try:
                             # Await cancellation to suppress warnings about uncaught exceptions (CancelledError)
                             await asyncio.wait_for(task, timeout=0.5)
                         except (asyncio.TimeoutError, asyncio.CancelledError):
                             pass # Expected outcomes for a cancelled task

                # --- Act Based on Outcome ---
                # Check again if shutdown happened during the wait or result processing
                if shutdown_triggered_during_wait or self._shutdown_event.is_set():
                    logger.info("Shutdown detected after wait/menu handling, breaking loop.")
                    self._running = False # Ensure flag is set to break the while loop
                    break # Exit the loop

                # If choice is None at this point, it means shutdown happened or an error occurred in menu handling
                # The 'continue' above handles the error case, so if we reach here with choice is None,
                # it's likely shutdown was signalled, which is caught by the previous if block.
                # This check is a bit redundant if shutdown handling is robust, but harmless.
                if choice is None:
                    # If we somehow get here without choice and not shutting down, log warning and continue
                    if self._running:
                        logger.warning("Menu returned no choice, but shutdown was not signalled. Continuing loop.")
                    continue # Continue to next loop iteration (will likely break immediately if _running is False)

                # --- Process Valid Menu Choice ---
                await self.process_menu_choice(choice)

            except Exception as loop_err:
                # Catch any other unexpected errors within the main loop logic itself (outside menu/action handling)
                logger.critical(f"Unexpected critical error in main run loop: {loop_err}", exc_info=True)
                self.ui.print_error(f"A critical error occurred in the main application loop: {loop_err}")
                # Trigger shutdown on critical loop errors
                if not self._shutdown_event.is_set():
                    self._create_task(self.shutdown(signal="MainLoopCriticalError", exit_code=1), name="ShutdownOnLoopError")
                # Break the loop immediately after attempting shutdown
                self._running = False # Ensure loop terminates
                break # Exit the loop

        logger.info("Main run loop finished.")


    async def process_menu_choice(self, choice: str):
        """Determines and executes the action based on the menu choice, handling disabled options."""
        # Check client status to determine which actions are available
        # Use the client's internal flags for robustness
        is_authenticated = self.exchange_client is not None and self.exchange_client._initialized

        # Define all possible menu options: (Display Text, Action Method Coroutine, Requires Auth Flag)
        # Use None for method if it requires auth but client isn't ready, or for Exit/special cases.
        # The order here dictates the menu order.
        all_menu_options: List[Tuple[str, Optional[Callable], bool]] = [
            ("Place Order", self.place_order_menu, True),
            ("View Account Balance", self.view_balance, True),
            ("View Open Orders", self.view_open_orders, True),
            ("View Open Positions", self.view_open_positions, True),
            ("Check Current Price", self.check_price_menu, True), # Needs client for fetch_ticker
            ("Technical Analysis", self.technical_analysis_menu, True), # Needs client for fetch_ohlcv
            ("Settings", self.settings_menu, False), # Settings always available
            ("Exit", None, False) # Special case handled explicitly below
        ]

        # Build a map from the choice index (string '1', '2', etc.) to the action details
        # This map reflects the menu that was actually displayed by ui_display_menu_async
        # It also determines if the selected action is *currently* executable based on auth status.
        available_actions_map: Dict[str, Tuple[str, Optional[Callable], bool]] = {}
        current_menu_index = 1
        for text, method, requires_auth in all_menu_options:
             action_method = method # Default to the method
             # If the action requires authentication AND the client is NOT initialized,
             # set the method to None for this menu instance.
             if requires_auth and not is_authenticated:
                  action_method = None
             # Store the original text, the potentially None method, and the auth flag
             available_actions_map[str(current_menu_index)] = (text, action_method, requires_auth)
             current_menu_index += 1

        # --- Handle the User's Choice ---
        if choice in available_actions_map:
            # Retrieve the action details based on the user's numerical choice
            option_text, action_method, requires_auth = available_actions_map[choice]

            # Handle Exit explicitly first (it has action_method=None in our definition, but check text)
            if option_text == "Exit":
                logger.info("Exit option selected by user.")
                # Trigger shutdown if it's not already in progress
                if not self._shutdown_event.is_set(): # Avoid duplicate shutdown calls
                    self._create_task(self.shutdown(signal="Exit Menu", exit_code=0), name="ShutdownOnExitMenu") # Normal exit code 0
                self._running = False # Signal loop to stop immediately
                return # Exit this function, main loop will break

            # Handle actions requiring auth when the client is not ready (action_method will be None)
            # Note: The UI marks these as disabled, but a user could still try to select by number.
            if requires_auth and action_method is None:
                self.ui.print_error("This action requires initialized API credentials and a connection.")
                self.ui.print_warning("Please check your .env file, restart the terminal, or check logs for connection errors.")
                # Pause for user to read the message using the async helper
                await self.ui_wait_for_enter_async()
                return # Return to main menu loop

            # Execute the action if it's available and the method is defined (action_method is not None)
            # Settings is available regardless of auth, other methods require auth and client initialized
            if action_method:
                action_name = option_text # Use menu text as a descriptive name for logging/errors
                logger.info(f"Executing action: {action_name}")
                # Call the action handler wrapper, which handles common exceptions
                await self.handle_action(action_method, action_name)
            else:
                # This case should theoretically not be reachable if Exit is handled and auth check is done correctly.
                # Log an internal error if it occurs.
                logger.error(f"Internal Error: Reached code for unavailable action '{option_text}' (choice {choice}). Should have been handled by auth check or exit logic.")
                self.ui.print_error(f"Selected action '{option_text}' is unexpectedly unavailable or not implemented.")
                # Pause for user to read
                await self.ui_wait_for_enter_async()
        else:
             # This should also not happen due to menu validation in ui.display_menu
             logger.error(f"Invalid menu choice '{choice}' received in process_menu_choice. This indicates a logic error or unexpected input flow.")
             self.ui.print_error(f"Internal Error: Invalid menu choice received: {choice}")
             # Pause briefly
             await asyncio.sleep(1)


    async def handle_action(self, action_coro: Callable[[], Any], action_name: str):
        """Executes a given menu action coroutine and handles common exceptions gracefully."""
        try:
            # Execute the specific action coroutine (e.g., self.view_balance())
            await action_coro()
            # Await for user acknowledgement is now typically handled *within* each action method
            # after displaying results, using ui_wait_for_enter_async().
            # So, no universal wait here.

        # --- Specific CCXT Error Handling (Catch most specific first) ---
        # Re-raise these specific CCXT errors from the client methods
        except ccxt.AuthenticationError as e:
             self.ui.print_error(f"Authentication Error during '{action_name}': {e}")
             self.ui.print_warning("Check API keys/secrets in .env and permissions on Bybit.")
             await self.ui_wait_for_enter_async() # Pause for user to read
        except ccxt.PermissionDenied as e:
             self.ui.print_error(f"Permission Denied during '{action_name}': {e}")
             self.ui.print_warning("API key lacks necessary permissions for this action (e.g., Trade, Read Wallet).")
             await self.ui_wait_for_enter_async()
        except ccxt.AccountNotEnabled as e: # Or similar if applicable
             self.ui.print_error(f"Account Issue during '{action_name}': {e}")
             self.ui.print_warning("Account may not be enabled for Futures/Contracts or requires setup on Bybit.")
             await self.ui_wait_for_enter_async()
        except ccxt.InsufficientFunds as e:
             self.ui.print_error(f"Insufficient Funds during '{action_name}': {e}")
             self.ui.print_warning("Check available balance, margin mode, and order cost.")
             await self.ui_wait_for_enter_async()
        except ccxt.InvalidOrder as e:
             # Log full error for details (already done in client.place_order), show simplified message + hint
             self.ui.print_error(f"Invalid Order parameters during '{action_name}'.")
             self.ui.print_warning(f"Details: {e}. Check symbol, size, price, tick/step sizes, SL/TP values, reduce-only conflicts, etc. See logs for full error.")
             await self.ui_wait_for_enter_async()
        except ccxt.BadRequest as e: # Catch issues like invalid params, sometimes account type issues
             logger.error(f"BadRequest during '{action_name}': {e}", exc_info=False) # Less verbose logging for common errors
             self.ui.print_error(f"Bad Request during '{action_name}': {e}.")
             self.ui.print_warning("Check parameters sent, symbol status, or potentially account type/settings.")
             await self.ui_wait_for_enter_async()
        except ccxt.BadSymbol as e:
             self.ui.print_error(f"Invalid Symbol during '{action_name}': {e}")
             self.ui.print_warning("Ensure symbol exists on Bybit USDT Perpetual market and is formatted correctly (e.g., BTC/USDT).")
             await self.ui_wait_for_enter_async()
        except ccxt.RequestTimeout as e:
             self.ui.print_error(f"Request Timeout during '{action_name}': {e}")
             self.ui.print_warning("Network connection to Bybit timed out. Check connection or try again later.")
             await self.ui_wait_for_enter_async()
        except ccxt.NetworkError as e: # Includes DNS issues, connection refused, etc. (parent of Timeout)
             self.ui.print_error(f"Network Error during '{action_name}': {e}")
             self.ui.print_warning("Could not reach Bybit. Check internet connection and Bybit status.")
             await self.ui_wait_for_enter_async()
        except ccxt.ExchangeNotAvailable as e: # Specific exchange downtime
             self.ui.print_error(f"Exchange Not Available during '{action_name}': {e}")
             self.ui.print_warning("Bybit might be down for maintenance or experiencing issues.")
             await self.ui_wait_for_enter_async()
        except ccxt.DDoSProtection as e: # Exchange rate limiting
             self.ui.print_error(f"Rate Limit / DDoS Protection during '{action_name}': {e}")
             self.ui.print_warning("You might be sending too many requests. Please wait a moment before trying again.")
             await asyncio.sleep(1) # Wait briefly before allowing user to press enter
             await self.ui_wait_for_enter_async()
        except ccxt.ExchangeError as e: # Catch-all for other specific exchange-reported errors
             logger.error(f"ExchangeError during '{action_name}': {e}{self.ui.colors['reset']}", exc_info=True)
             self.ui.print_error(f"An Exchange Error occurred during '{action_name}': {e}")
             self.ui.print_warning("This is an error reported by the Bybit API. Check details.")
             await self.ui_wait_for_enter_async()

        # --- Application / Input / Validation Errors ---
        except ConnectionError as e: # Raised by _check_initialized or client init failures propagated
             self.ui.print_error(f"Connection Error during '{action_name}': {e}")
             self.ui.print_warning("Client might be offline, uninitialized, or encountered an issue (e.g., account type mismatch). Check logs.")
             await self.ui_wait_for_enter_async()
        except ValueError as e: # Catch validation errors etc. from get_input or data processing (e.g., TA, pivots)
             self.ui.print_error(f"Input or Data Error during '{action_name}': {e}")
             # Log less verbose stack for typical value errors
             logger.error(f"Input/Data ValueError during '{action_name}': {e}", exc_info=False)
             await self.ui_wait_for_enter_async()
        except TypeError as e: # Catch unexpected type issues, e.g., in data processing
             self.ui.print_error(f"Type Error during '{action_name}': {e}")
             logger.error(f"TypeError during '{action_name}': {e}{self.ui.colors['reset']}", exc_info=True)
             await self.ui_wait_for_enter_async()

        # --- Interruptions ---
        except (EOFError, KeyboardInterrupt) as e:
             # These are raised by the async input helpers if encountered *during* an action
             logger.warning(f"'{type(e).__name__}' caught during action '{action_name}'. Initiating shutdown.")
             # Trigger shutdown if it's not already in progress
             if not self._shutdown_event.is_set(): # Prevent duplicate calls
                 # Use exit code 1 for interrupt during operation
                 self._create_task(self.shutdown(signal=type(e).__name__, exit_code=1), name=f"ShutdownOnActionInterrupt")
             self._running = False # Signal main loop to stop
             # Don't wait for enter, let shutdown proceed / main loop break immediately

        # --- Catch-all for Unexpected Errors ---
        except Exception as e:
            # Catch any other unexpected errors during the action's execution
            logger.critical(f"Unhandled critical error in action '{action_name}': {e}{self.ui.colors['reset']}", exc_info=True)
            self.ui.print_error(f"An unexpected critical error occurred in '{action_name}': {e}")
            self.ui.print_warning(f"Check '{log_file}' for detailed traceback.")
            await self.ui_wait_for_enter_async()
            # Consider if critical errors in actions should trigger shutdown?
            # For now, just report and return to menu unless it's an interrupt.


    # --- Async Input Helpers (Using run_in_executor) ---
    # These wrappers run the blocking UI functions in a separate thread pool
    # managed by asyncio's default executor, preventing the UI from blocking
    # the main async event loop. They also handle propagating interrupts.

    async def ui_display_menu_async(self) -> str:
        """Displays the main menu and gets choice asynchronously via executor."""
        loop = asyncio.get_running_loop()

        # Determine available actions based on client status for dynamic menu display
        is_authenticated = self.exchange_client is not None and self.exchange_client._initialized

        # Define menu options text and auth requirement (matches process_menu_choice)
        all_menu_options_text = [
            ("Place Order", True), ("View Account Balance", True), ("View Open Orders", True),
            ("View Open Positions", True), ("Check Current Price", True),
            ("Technical Analysis", True), # Requires client for data fetching
            ("Settings", False), ("Exit", False)
        ]

        # Build the list of options to actually display, marking disabled ones
        current_menu_display_options = []
        for text, requires_auth in all_menu_options_text:
             if requires_auth and not is_authenticated:
                  # Append disabled marker using theme colors (dim style)
                  # Ensure the main text uses the standard menu option color before the dim marker
                  disabled_text = f"{self.ui.colors['menu_option']}{text} {self.ui.colors['dim']}(Connection Required){self.ui.colors['reset']}"
                  current_menu_display_options.append(disabled_text)
             else:
                  # Option is available, just use the text
                  current_menu_display_options.append(text)

        # Construct menu title, adding Testnet/Limited Mode indicators
        menu_title = "Pyrmethus Bybit Terminal"
        if self.credentials and self.credentials.testnet:
             menu_title += f" {self.ui.colors['accent']}(Testnet){self.ui.colors['reset']}" # Use accent for testnet
        # Only show limited mode if NOT authenticated
        if not is_authenticated:
             # Add warning color/text for limited mode
             menu_title += f" {self.ui.colors['warning']}(Limited Mode){self.ui.colors['reset']}"
             # Add a hint if it's due to missing credentials vs connection failure
             if self.credentials is None:
                  menu_title += f" {self.ui.colors['dim']}(Check .env){self.ui.colors['reset']}"
             else:
                  menu_title += f" {self.ui.colors['dim']}(Check Connection){self.ui.colors['reset']}"


        # Wrap the blocking UI function call with its arguments using partial
        # Pass the dynamically generated options list
        func_call = partial(self.ui.display_menu, menu_title, current_menu_display_options, "Select your spell")

        # Run the blocking function in the default executor (thread pool)
        try:
            choice = await loop.run_in_executor(None, func_call)
            return choice
        except (EOFError, KeyboardInterrupt) as e:
            # Interruption occurred within the executor thread (input)
            logger.warning(f"{type(e).__name__} caught in executor wrapper for display_menu.")
            # Re-raise the specific exception to be caught by the caller (run loop)
            raise e
        except Exception as e:
            # Catch unexpected errors from within the ui.display_menu function itself
            logger.error(f"Unexpected error in display_menu executor task: {e}{self.ui.colors['reset']}", exc_info=True)
            # Propagate unexpected errors, wrapping them for clarity
            raise RuntimeError(f"Failed to display menu via executor: {e}") from e


    async def ui_get_input_async(self, prompt: str, **kwargs) -> Any:
        """Asynchronously gets validated user input using executor."""
        loop = asyncio.get_running_loop()
        # Wrap the call to self.ui.get_input with its arguments using partial
        # Pass prompt directly, other args via kwargs dictionary
        func_call = partial(self.ui.get_input, prompt, **kwargs)
        try:
            # Run the blocking input function in the executor
            return await loop.run_in_executor(None, func_call)
        except (EOFError, KeyboardInterrupt) as e:
             # Interruption occurred within the executor thread (input)
             logger.warning(f"{type(e).__name__} caught in executor wrapper for get_input (prompt: '{prompt}').")
             # Re-raise to be caught by handle_action or the calling function
             raise e
        except Exception as e:
             # Catch unexpected errors from within the ui.get_input function itself
             logger.error(f"Unexpected error in get_input executor task (prompt: '{prompt}'): {e}{self.ui.colors['reset']}", exc_info=True)
             raise RuntimeError(f"Failed to get user input via executor: {e}") from e


    async def ui_wait_for_enter_async(self, prompt: str = "Press Enter to continue..."):
        """Asynchronously waits for Enter key using executor."""
        loop = asyncio.get_running_loop()
        # Wrap the call using partial
        func_call = partial(self.ui.wait_for_enter, prompt)
        try:
            # Run the blocking input function in the executor
            await loop.run_in_executor(None, func_call)
        except (EOFError, KeyboardInterrupt) as e:
             # Interruption occurred within the executor thread (input)
             logger.warning(f"{type(e).__name__} caught in executor wrapper for wait_for_enter.")
             # Re-raise to be caught by handle_action or the calling function
             raise e
        except Exception as e:
             # Catch unexpected errors from within the ui.wait_for_enter function itself
             logger.error(f"Unexpected error in wait_for_enter executor task: {e}{self.ui.colors['reset']}", exc_info=True)
             raise RuntimeError(f"Failed waiting for Enter via executor: {e}") from e


    # --- Input Validation Helpers ---

    def _validate_symbol(self, symbol: str) -> Union[bool, str]:
        """Validates symbol format (CCXT standard BASE/QUOTE) and checks against loaded markets if available."""
        if not isinstance(symbol, str) or not symbol:
            return "Symbol cannot be empty."
        symbol_input_cleaned = symbol.strip() # Keep original case for potential ID check later
        symbol_upper = symbol_input_cleaned.upper() # Uppercase for format check

        # Basic format check: Allow letters, numbers, and exactly one slash separating non-empty parts.
        # Disallow leading/trailing slashes. Allow hyphens if needed.
        if '/' in symbol_upper:
             parts = symbol_upper.split('/')
             if len(parts) != 2 or not parts[0] or not parts[1]:
                  return f"Invalid format '{symbol}'. Use BASE/QUOTE (e.g., BTC/USDT)."
             # Further check parts only contain letters, numbers, hyphen?
             if not re.match(r"^[A-Z0-9-]+$", parts[0]) or not re.match(r"^[A-Z0-9-]+$", parts[1]):
                  return f"Invalid characters in symbol parts '{symbol}'. Use letters, numbers, hyphen, and a single slash."

        # If no slash, check if it looks like a concatenated format (e.g., BTCUSDT)
        elif re.match(r"^[A-Z0-9-]+$", symbol_upper): # Allow letters, numbers, hyphen
             # Looks like a potential concatenated symbol or single asset. Will try to format later.
             pass
        else:
             return f"Invalid symbol format '{symbol}'. Use BASE/QUOTE (e.g., BTC/USDT) or concatenated form (e.g., BTCUSDT)."


        # Try to normalize to CCXT standard format (BASE/QUOTE) for market check
        ccxt_symbol = self._get_ccxt_symbol(symbol_input_cleaned) # Use the cleaned input string
        # If _get_ccxt_symbol couldn't produce a plausible format, it might return the original or empty string
        if not ccxt_symbol:
             # This case is less likely after the initial format check, but defensive
             return f"Could not normalize symbol '{symbol}' to a standard format."

        # Validate against loaded markets if client is ready
        if self.exchange_client and self.exchange_client._markets_loaded:
           # Check if the exchange object and markets dictionary exist
           if not self.exchange_client.exchange or not self.exchange_client.exchange.markets:
                logger.warning(f"{self.ui.colors['warning']}Markets object missing or empty during symbol validation. Skipping market existence check.{self.ui.colors['reset']}")
                # Cannot validate further, assume okay for now based on format
                return True

           # Check if the normalized symbol exists in the loaded markets
           if ccxt_symbol not in self.exchange_client.exchange.markets:
               # Try to provide helpful suggestions based on available active USDT perpetuals
               available_symbols = list(self.exchange_client.exchange.markets.keys())
               # Filter for active USDT perpetuals specifically
               usdt_swap_symbols = sorted([
                   s for s, m in self.exchange_client.exchange.markets.items()
                   if isinstance(s, str) and m.get('swap') and m.get('linear') and m.get('quote') == 'USDT' and m.get('active')
               ])
               suggestions = []
               # Try matching based on base currency if user entered concatenated form or partial match
               # Remove potential '/USDT' or 'USDT' from the end for base guess
               user_base_guess = symbol_upper.replace('/USDT', '').replace('USDT', '')
               if user_base_guess:
                   # Find symbols starting with the base guess or containing it
                   suggestions = [s for s in usdt_swap_symbols if s.startswith(user_base_guess + '/') or user_base_guess in s.replace('/', '')]

               # If still no suggestions, provide general examples from USDT swaps
               if not suggestions and usdt_swap_symbols:
                    # Limit the number of general examples
                    suggestions = usdt_swap_symbols[:10] # Take up to 10 examples

               suggestion_str = ""
               if suggestions:
                    # Limit suggestions shown to a reasonable number
                    suggestion_str = f" Did you mean one of these active USDT swaps? {', '.join(suggestions[:5])}{'...' if len(suggestions) > 5 else ''}"
               elif available_symbols: # Fallback to any available symbol examples if no USDT swaps found (unlikely)
                    suggestion_str = f" Note: Only active USDT Perpetual swaps are supported. Available market symbols start with: {', '.join(available_symbols[:5])}..."

               return f"Symbol '{ccxt_symbol}' not found or not an active USDT Perpetual swap.{suggestion_str}"
           else:
                # Symbol found, double check it's an active USDT perpetual swap
                market_info = self.exchange_client.exchange.markets[ccxt_symbol]
                if not (market_info.get('swap') and market_info.get('linear') and market_info.get('quote') == 'USDT' and market_info.get('active')):
                    return f"Symbol '{ccxt_symbol}' exists but is not an active USDT Perpetual swap market."

        # If markets are NOT loaded, only perform basic format validation
        # The initial regex check handles this.
        return True # Passed basic format checks and market check (if performed)

    def _get_ccxt_symbol(self, user_input: str) -> str:
        """Converts user input (e.g., BTCUSDT or btc/usdt) to CCXT standard format (BTC/USDT). Tries matching market IDs and BASEQUOTE."""
        if not isinstance(user_input, str): return "" # Handle non-string input
        symbol = user_input.strip() # Keep original case for ID matching
        symbol_upper = symbol.upper() # Uppercase for other checks

        # If already in standard format, return directly
        if '/' in symbol_upper and re.match(r"^[A-Z0-9-]+/[A-Z0-9-]+$", symbol_upper):
             return symbol_upper # Return uppercase standard format

        # If markets are loaded, try to find the CCXT symbol matching the input more reliably
        if self.exchange_client and self.exchange_client._markets_loaded and self.exchange_client.exchange and self.exchange_client.exchange.markets:
             markets = self.exchange_client.exchange.markets
             # 1. Check if input matches an exchange-specific ID ('id' field) for an active USDT swap
             # Use the original case input for ID matching as IDs are case-sensitive on some exchanges
             for market_symbol, market_data in markets.items():
                 if market_data.get('id') == symbol: # Case-sensitive ID check
                     # Ensure it's an active USDT perpetual swap (linear)
                     if market_data.get('swap') and market_data.get('linear') and market_data.get('quote') == 'USDT' and market_data.get('active'):
                        logger.debug(f"Normalized symbol '{symbol}' to '{market_symbol}' based on matching market ID (Active USDT Swap).")
                        return market_symbol # Return the CCXT standard symbol from market data
                     else:
                         logger.debug(f"Symbol '{symbol}' matched market ID '{market_symbol}' but it's not an active USDT Perpetual Swap. Skipping ID match.")
                         # Continue searching in case of other matches

             # 2. Check if input matches concatenated BASEQUOTE for an active USDT swap
             # Use uppercase input for concatenated symbol check
             possible_matches = []
             for market_symbol, market_info in markets.items():
                  base = market_info.get('base')
                  quote = market_info.get('quote')
                  is_swap = market_info.get('swap', False)
                  is_linear = market_info.get('linear', False)
                  is_active = market_info.get('active', False)

                  if base and quote and is_swap and is_linear and is_active and quote == 'USDT':
                      if symbol_upper == f"{base}{quote}".upper(): # Compare uppercase versions
                          possible_matches.append(market_symbol)

             if len(possible_matches) == 1:
                  formatted = possible_matches[0]
                  logger.debug(f"Normalized symbol '{symbol}' to '{formatted}' based on concatenated BASEQUOTE (Active USDT Swap).")
                  return formatted
             elif len(possible_matches) > 1:
                  # If multiple matches (unlikely but possible), pick the first and warn
                  logger.warning(f"{self.ui.colors['warning']}Ambiguous symbol '{symbol}'. Multiple active USDT swaps match concatenated form: {possible_matches}. Returning first match: {possible_matches[0]}{self.ui.colors['reset']}")
                  return possible_matches[0] # Return first match but log warning

        # 3. Generic guess if markets not loaded or no specific match found
        # Assume it ends with USDT and try to split (less reliable)
        # Use uppercase input for this guess
        if symbol_upper.endswith("USDT") and len(symbol_upper) > 4:
            base = symbol_upper[:-4]
            formatted = f"{base}/USDT"
            logger.debug(f"Normalized symbol '{symbol}' to '{formatted}' (generic USDT guess - market check recommended).")
            return formatted

        # If no normalization rule applied, return the original uppercase input (validation function will likely catch it if markets are loaded)
        logger.warning(f"{self.ui.colors['warning']}Could not reliably normalize symbol '{symbol}' to BASE/USDT swap format using market data or patterns. Using original uppercase input '{symbol_upper}'.{self.ui.colors['reset']}")
        return symbol_upper

    def _validate_side(self, value: Any) -> Union[bool, str]:
        """Validates order side (buy/sell). Expects string input."""
        if not isinstance(value, str) or value.strip().lower() not in ['buy', 'sell']:
            return "Invalid side. Must be 'buy' or 'sell'."
        return True

    def _validate_order_type(self, value: Any) -> Union[bool, str]:
        """Validates order type (Market/Limit). Expects string input. Case-insensitive check, returns error message or True."""
        # Expand supported types if exchange/CCXT allows more easily (e.g., Stop, StopLimit) via params later
        supported_types = ['market', 'limit'] # Keep main types simple for direct input
        if not isinstance(value, str) or value.strip().lower() not in supported_types:
            supported_str = ', '.join(t.capitalize() for t in supported_types)
            return f"Invalid order type. Use {supported_str}."
        return True

    def _validate_float(self, value: Any) -> Union[bool, str]:
        """Validates if the value is a valid finite float."""
        if value is None or (isinstance(value, str) and not value.strip()):
             return "Input cannot be empty."
        try:
            num_value = float(value) # Attempt conversion
            if not np.isfinite(num_value): return "Value must be a finite number (not infinity or NaN)."
            return True
        except (ValueError, TypeError):
            return "Input must be a valid number (e.g., 0.5, 100)."

    def _validate_positive_float(self, value: Any) -> Union[bool, str]:
        """Validates if the value is a float strictly greater than zero."""
        # First, check if it's a valid float
        float_check = self._validate_float(value)
        if float_check is not True: return float_check # Return the error message from float validation

        num_value = float(value) # Already converted and checked finite
        # Use a small tolerance for comparison > 0 if needed, but direct check is usually fine
        if num_value <= 0: return "Value must be positive (greater than 0)."
        return True

    def _validate_non_negative_float(self, value: Any) -> Union[bool, str]:
        """Validates if the value is a float greater than or equal to zero."""
        # First, check if it's a valid float
        float_check = self._validate_float(value)
        if float_check is not True: return float_check # Return the error message from float validation

        num_value = float(value) # Already converted and checked finite
        if num_value < 0: return "Value cannot be negative."
        return True

    def _validate_percentage(self, value: Any) -> Union[bool, str]:
        """Validate percentage, typically > 0 and <= 100 for trailing stops, checking Bybit limits."""
        # First, ensure it's a positive float
        result = self._validate_positive_float(value)
        if result is not True: return result # Return error message from positive float check

        num_value = float(value)
        # Check against typical exchange limits for trailing stops (e.g., Bybit's 0.01% to 10% or similar)
        # These limits might vary, using a reasonable range here.
        min_percent = 0.01 # Bybit's typical minimum trail percentage
        max_percent = 50.0  # Setting a generous upper limit (Bybit's actual max might be lower, e.g., 10% or 20%)

        if num_value < min_percent:
            return f"Percentage seems too small (min is typically around {min_percent}%)."
        if num_value > max_percent:
            return f"Percentage seems too large (max is typically around {max_percent}% or less)."
        return True

    def _validate_integer(self, value: Any) -> Union[bool, str]:
         """Validates if the value is a valid integer."""
         if value is None or (isinstance(value, str) and not value.strip()):
              return "Input cannot be empty."
         try:
             # Allow float input but check if it's a whole number
             num_value = float(value)
             if not np.isfinite(num_value): return "Value must be a finite number."
             if not num_value.is_integer(): return "Value must be a whole number (integer)."
             return True # It's a finite integer
         except (ValueError, TypeError):
             return "Input must be a valid integer."

    def _validate_integer_range(self, value: Any, min_val: int, max_val: int) -> Union[bool, str]:
         """Validates if the value is an integer within a specific range (inclusive)."""
         # First, validate it's an integer
         int_check = self._validate_integer(value)
         if int_check is not True: return int_check # Return the error message from integer validation

         num_value = int(float(value)) # Convert to int after checking it's a whole number float
         if min_val <= num_value <= max_val:
              return True
         else:
              return f"Value must be between {min_val} and {max_val} (inclusive)."


    def _validate_timeframe(self, value: Any) -> Union[bool, str]:
        """Validates CCXT timeframe format and checks against loaded exchange timeframes if available."""
        if not isinstance(value, str) or not value:
            return "Timeframe cannot be empty."
        timeframe_in = value.strip().lower() # Normalize to lowercase for checks

        # Regex for common CCXT styles (e.g., 1m, 5m, 1h, 4h, 1d, 1w, 1M)
        # Allows 1-3 digits for number part, single char [mhdwyM] for unit.
        # Also allow Bybit specific single letters D, W, M for input flexibility.
        # Add anchors ^ and $ to ensure the whole string matches the pattern
        if not re.match(r"^\d{1,3}[mhdwyM]$", timeframe_in) and not re.match(r"^[dwm]$", timeframe_in):
             return "Invalid timeframe format. Use CCXT style (e.g., '1m', '15m', '1h', '4h', '1d', '1W', '1M') or Bybit single letters (D, W, M)."

        # Map Bybit single letters to CCXT standard for internal consistency and exchange check
        tf_map = {'d': '1d', 'w': '1w', 'm': '1M'}
        # Use the lowercase input for mapping key lookup, ensure result is lowercase standard
        timeframe_check = tf_map.get(timeframe_in, timeframe_in)

        # Optional: Check against exchange.timeframes if client and markets are loaded
        # Access the client instance indirectly via config_manager or terminal if needed
        # For now, this validation helper doesn't have direct access to the client.
        # A more robust solution would be to pass the client or available timeframes.
        # Let's keep this basic format validation and rely on the client's fetch_ohlcv to raise ExchangeError for unsupported TFs.

        # If we had access to the client:
        # if self.exchange_client and self.exchange_client._markets_loaded and self.exchange_client.exchange:
        #    if hasattr(self.exchange_client.exchange, 'timeframes') and isinstance(self.exchange_client.exchange.timeframes, dict):
        #        available_tfs = self.exchange_client.exchange.timeframes
        #        if timeframe_check not in available_tfs:
        #            # Add suggestions...
        #            return f"Timeframe '{timeframe_check}' may not be supported by the exchange. Available: ..."
        #    else:
        #         logger.warning("Exchange timeframes data not available. Skipping timeframe support check.")


        return True # Passed format validation (exchange support check skipped here)


    # --- Menu Actions ---
    # These methods are called by process_menu_choice and wrapped by handle_action for error handling.
    # They use the ui_..._async helpers for user interaction.

    async def place_order_menu(self):
        """Handles the user interaction logic for placing an order."""
        self.ui.display_header("Place Order Spell")

        # --- Get Order Details ---
        # Symbol - Use default from config, validate format and existence
        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        # Pass the validation function to the async input helper
        symbol_input = await self.ui_get_input_async("Enter symbol (e.g., BTC/USDT or BTCUSDT)", default=default_symbol, validation_func=self._validate_symbol)
        if symbol_input is None: return # User cancelled or error propagated by get_input

        # Normalize symbol after validation passes. _get_ccxt_symbol handles different input formats.
        symbol = self._get_ccxt_symbol(symbol_input)

        # Side - Use validation function
        side_input = await self.ui_get_input_async("Enter side (buy/sell)", validation_func=self._validate_side)
        if side_input is None: return
        side = side_input.strip().lower() # Normalize for CCXT

        # Order Type - Use default from config, validate type
        default_order_type = self.config_manager.config.get("default_order_type", "Limit")
        # Pass validation function to the async input helper
        order_type_input = await self.ui_get_input_async("Enter order type (Market/Limit)", default=default_order_type, validation_func=self._validate_order_type)
        if order_type_input is None: return
        order_type = order_type_input.strip().lower() # Use lowercase for CCXT

        # Amount (Quantity) - Use float input type and positive validation
        amount = await self.ui_get_input_async("Enter quantity (in base currency, e.g., BTC amount)", input_type=float, validation_func=self._validate_positive_float)
        if amount is None: return # Exit if input failed validation or was cancelled

        # Price (for Limit orders) - Use float input type and positive validation
        price: Optional[float] = None
        if order_type == 'limit':
            price = await self.ui_get_input_async("Enter limit price", input_type=float, validation_func=self._validate_positive_float)
            if price is None: return # Price required for limit order, exit if cancelled/error

        # --- Optional Parameters ---
        params = {} # Dictionary for CCXT standard order parameters + exchange specifics
        self.ui.print_info("\nConfigure optional parameters (leave blank or 0 for none):")

        # Stop Loss (Trigger Price) - Use float input type and non-negative validation
        # Required=False allows blank input, which defaults to "0" via default param
        sl_price = await self.ui_get_input_async("Stop Loss trigger price (0 for none)", default="0", required=False, input_type=float, validation_func=self._validate_non_negative_float)
        if sl_price is not None and sl_price > 0: # Only add to params if greater than 0
            params['stopLossPrice'] = sl_price # CCXT standard key for trigger price
            # Optional: Add Bybit V5 specific trigger type input if needed (e.g., slTriggerBy)
            # sl_trigger = await self.ui_get_input_async("SL Trigger By (MarkPrice/LastPrice/IndexPrice)", default="MarkPrice", required=False)
            # if sl_trigger: params['slTriggerBy'] = sl_trigger.capitalize() # Add to params if provided
            self.ui.print_info(f"Stop Loss trigger set to {sl_price}")

        # Take Profit (Trigger Price) - Use float input type and non-negative validation
        tp_price = await self.ui_get_input_async("Take Profit trigger price (0 for none)", default="0", required=False, input_type=float, validation_func=self._validate_non_negative_float)
        if tp_price is not None and tp_price > 0: # Only add to params if greater than 0
            params['takeProfitPrice'] = tp_price # CCXT standard key for trigger price
            # Optional: Add Bybit V5 specific trigger type input if needed (e.g., tpTriggerBy)
            # tp_trigger = await self.ui_get_input_async("TP Trigger By (MarkPrice/LastPrice/IndexPrice)", default="MarkPrice", required=False)
            # if tp_trigger: params['tpTriggerBy'] = tp_trigger.capitalize() # Add to params if provided
            self.ui.print_info(f"Take Profit trigger set to {tp_price}")

        # Trailing Stop (Percentage) - Use float input type and percentage validation
        # Required=False allows blank input, which defaults to "0" via default param
        trail_percent = await self.ui_get_input_async("Trailing Stop percentage (e.g., 0.5 for 0.5%, min ~0.01, 0 for none)", default="0", required=False, input_type=float, validation_func=self._validate_percentage)
        if trail_percent is not None and trail_percent > 0: # Only add to params if greater than 0
            # Use standard CCXT param 'trailingPercent' (expects percentage value, e.g., 0.5 for 0.5%)
            params['trailingPercent'] = trail_percent
            # Note: Bybit V5 might need activation price ('activePrice' param) or other params if CCXT mapping fails.
            # Consider adding input for activation price if issues arise or more control is needed.
            # active_price = await self.ui_get_input_async("Trailing Stop activation price (optional, 0 for none)", default="0", required=False, input_type=float, validation_func=self._validate_non_negative_float)
            # if active_price is not None and active_price > 0: params['activePrice'] = active_price
            self.ui.print_info(f"Trailing Stop set at {trail_percent}%. Activation details depend on exchange defaults if not specified.")


        # Reduce Only - Use simple string input and check
        reduce_only_input = await self.ui_get_input_async("Set as Reduce Only? (yes/no)", default="no", required=False, input_type=str, validation_func=lambda s: s.strip().lower() in ['yes', 'no'] or "Must be 'yes' or 'no'")
        if reduce_only_input and reduce_only_input.strip().lower() == 'yes':
             params['reduceOnly'] = True
             self.ui.print_info("Order set to Reduce Only.")

        # --- Confirmation ---
        # Build confirmation message dynamically
        confirm_parts = [f"Confirm: Place {order_type.upper()} {side.upper()} order for {amount} {symbol}"]
        if price is not None: confirm_parts.append(f"at price {price}")
        # Check if Limit order but price is None (should be caught by validation, but defensive)
        elif order_type == 'limit' and price is None: confirm_parts.append(f"{self.ui.colors['error']}(ERROR: No price for Limit order!){self.ui.colors['reset']}")
        if params.get('stopLossPrice') is not None: confirm_parts.append(f"with SL @ {params['stopLossPrice']}")
        if params.get('takeProfitPrice') is not None: confirm_parts.append(f"with TP @ {params['takeProfitPrice']}")
        if params.get('trailingPercent') is not None: confirm_parts.append(f"with Trail {params['trailingPercent']}%")
        if params.get('reduceOnly'): confirm_parts.append("(Reduce Only)")
        confirm_msg = " ".join(confirm_parts) + "?"

        print(f"\n{self.ui.colors['warning']}{confirm_msg}{self.ui.colors['reset']}")
        # Use simple string input for confirmation
        confirm = await self.ui_get_input_async("Type 'yes' to confirm", default="no", required=False)

        # Check if confirmation input is 'yes' (case-insensitive)
        if confirm is None or not isinstance(confirm, str) or confirm.strip().lower() != "yes":
            self.ui.print_warning("Order spell cancelled by user.")
            # Pause for user to read before returning to menu
            await self.ui_wait_for_enter_async()
            return

        # --- Place the Order via Client ---
        self.ui.print_info(f"{self.ui.colors['primary']}# Submitting order incantation...{self.ui.colors['reset']}")
        # Call the client method. Exceptions (like InsufficientFunds, InvalidOrder)
        # will be caught by the handle_action wrapper.
        # Pass the determined parameters dictionary
        result = await self.exchange_client.place_order(symbol, side, order_type, amount, price, params)

        # --- Display Result ---
        self.ui.print_success("Order submission attempt processed by exchange.")

        # Display selected details from the result dict using print_table for dict
        # Focus on key details returned by CCXT create_order structure
        # Include standard CCXT fields and potentially check 'info' for exchange specifics
        display_keys = ['id', 'clientOrderId', 'datetime', 'timestamp', 'status', 'symbol', 'type', 'side',
                        'price', 'amount', 'cost', 'average', 'filled', 'remaining', 'fee',
                        # Include standard trigger/trail keys if they were passed in params, CCXT might return them
                        'stopLossPrice', 'takeProfitPrice', 'trailingPercent', 'reduceOnly']


        display_result = {}
        for k in display_keys:
             value = result.get(k)
             # Only add to display if the value is not None
             if value is not None:
                 # Special formatting for specific keys
                 if k == 'fee' and isinstance(value, dict):
                      fee_info = value
                      cost = fee_info.get('cost')
                      currency = fee_info.get('currency')
                      # Use high precision for fee cost
                      display_result['Fee'] = f"{cost:.8f} {currency}" if cost is not None and currency else str(fee_info)
                 # Format datetime from string or timestamp
                 elif k == 'datetime' and value:
                      try: # Use pandas for robust parsing
                          dt_obj = pd.to_datetime(value)
                          display_result['Timestamp'] = dt_obj.strftime('%Y-%m-%d %H:%M:%S UTC')
                      except Exception: display_result['Timestamp'] = value # Fallback
                 elif k == 'timestamp' and value and 'Timestamp' not in display_result: # Use if datetime missing
                      try:
                           dt_obj = pd.to_datetime(value, unit='ms')
                           display_result['Timestamp'] = dt_obj.strftime('%Y-%m-%d %H:%M:%S UTC')
                      except Exception: display_result['Timestamp'] = value
                 # Format floats with appropriate precision
                 elif isinstance(value, (float, np.floating)):
                     # Use higher precision for crypto prices/amounts
                     precision = 8 if k in ['price', 'amount', 'cost', 'filled', 'remaining', 'average'] else 4
                     # Use standard Python f-string formatting
                     display_result[k.capitalize()] = f"{value:.{precision}f}"
                 # Handle boolean ReduceOnly for display
                 elif k == 'reduceOnly' and isinstance(value, bool):
                      display_result[k.capitalize()] = "Yes" if value else "No"
                 else:
                      # Capitalize key for display consistency for other types
                      display_result[k.capitalize()] = value

        # Check 'info' dict for potentially missing Bybit-specific details if needed
        if 'info' in result and isinstance(result['info'], dict):
             info_data = result['info']
             # Add Bybit specific trigger/trail info if not found in standard keys or for confirmation
             # Use Bybit V5 keys like 'stopLoss', 'takeProfit', 'trailingStop', 'reduceOnly', 'orderStatus'
             if 'Slprice' not in display_result and info_data.get('stopLoss'): display_result['SL Price (Info)'] = info_data['stopLoss']
             if 'Tpprice' not in display_result and info_data.get('takeProfit'): display_result['TP Price (Info)'] = info_data['takeProfit']
             # Bybit V5 uses 'trailingStop' for active trail value (often "0" if inactive)
             if 'Trailingpercent' not in display_result and info_data.get('trailingStop') and info_data['trailingStop'] != "0":
                  display_result['Trail Info (Info)'] = info_data['trailingStop']
             # Bybit V5 uses 'reduceOnly' boolean
             if 'Reduceonly' not in display_result and info_data.get('reduceOnly') is not None: display_result['Reduce Only (Info)'] = "Yes" if info_data['reduceOnly'] else "No"
             # Bybit V5 order status - might be more granular than CCXT status
             if 'Status' not in display_result and info_data.get('orderStatus'): display_result['Status (Info)'] = info_data['orderStatus']


        # Ensure essential fields like ID, Status, Symbol, Side, Type are present, even if N/A
        # Use .get() with default 'N/A' to handle missing keys robustly
        essential_keys = ['id', 'status', 'symbol', 'side', 'type']
        for k in essential_keys:
             if k.capitalize() not in display_result:
                 # Get the value from the original result dict, capitalize and add
                 val = result.get(k, 'N/A')
                 # Handle Side/Type capitalization if needed, though CCXT usually returns lowercase
                 if k in ['side', 'type']: val = str(val).capitalize()
                 display_result[k.capitalize()] = val

        # Reorder keys for better display if needed (optional, print_table iterates dict keys)
        # Example manual reordering:
        # ordered_display = {'Id': display_result.pop('Id', 'N/A')}
        # ordered_display['Timestamp'] = display_result.pop('Timestamp', 'N/A')
        # ordered_display['Status'] = display_result.pop('Status', 'N/A')
        # ordered_display['Symbol'] = display_result.pop('Symbol', 'N/A')
        # ordered_display['Side'] = display_result.pop('Side', 'N/A')
        # ordered_display['Type'] = display_result.pop('Type', 'N/A')
        # # Add remaining keys from display_result
        # ordered_display.update(display_result)
        # self.ui.print_table(ordered_display, ...)
        # For now, let print_table use the dict's iteration order.

        # Use print_table for dictionary display (handles formatting/colors/None)
        # Use higher precision default for crypto values
        self.ui.print_table(display_result, title="Order Submission Result Details", float_format='{:.8f}')
        await self.ui_wait_for_enter_async()


    async def view_balance(self):
        """Displays account balances (USDT Futures) using CCXT V5 handling."""
        self.ui.display_header("Account Balance Scrying (USDT Futures)")
        self.ui.print_info(f"{self.ui.colors['primary']}# Fetching balance essence...{self.ui.colors['reset']}")

        # Fetch balance (client method handles V5 params and parsing)
        balance_data = await self.exchange_client.fetch_balance() # Returns the 'USDT' dict or parsed equivalent

        if not balance_data or not isinstance(balance_data, dict):
            self.ui.print_warning("No detailed USDT balance data found or format unexpected.")
            # If fetch_balance returned something non-standard, log it
            if balance_data: logger.warning(f"Unexpected balance data structure received: {balance_data}")
            await self.ui_wait_for_enter_async()
            return

        # Extract data using standard CCXT keys ('free', 'used', 'total') expected from fetch_balance
        # Use .get() with default None initially, then format or set N/A
        free_balance = balance_data.get('free')
        used_margin = balance_data.get('used')
        total_equity = balance_data.get('total')

        # Extract PNL from standardized keys populated by fetch_balance
        # Use .get() with default None to handle cases where PNL might be missing
        unrealized_pnl = balance_data.get('unrealizedPnl')
        realized_pnl = balance_data.get('cumRealisedPnl') # Standard key populated by our client parsing

        # Prepare data for display table, format floats and handle None
        display_data = {
            "Total Equity": total_equity,
            "Available Balance": free_balance, # CCXT 'free' maps to available balance for futures/swap
            "Used Margin": used_margin,
            # Optionally add Wallet Balance if available from V5 parsing (added by client)
            "Wallet Balance (V5)": balance_data.get('v5_walletBalance'), # Might be None if not parsed
            "Unrealized PNL": unrealized_pnl, # Placeholder, formatted below
            "Realized PNL": realized_pnl,   # Placeholder, formatted below
        }

        # Format PNL with color, handle None values
        for pnl_key in ["Unrealized PNL", "Realized PNL"]:
            pnl_value = display_data.get(pnl_key)
            if pnl_value is not None:
                try:
                    pnl_float = float(pnl_value)
                    # Use appropriate color for PNL based on value
                    pnl_color = self.ui.colors['positive'] if pnl_float > 0 else self.ui.colors['negative'] if pnl_float < 0 else self.ui.colors['neutral']
                    # Format with color and 4 decimal places
                    display_data[pnl_key] = f"{pnl_color}{pnl_float:.4f}{self.ui.colors['reset']}"
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse value '{pnl_value}' for {pnl_key}.")
                    display_data[pnl_key] = f"{self.ui.colors['error']}Parse Error{self.ui.colors['reset']}"
            else:
                 # Explicitly show N/A if the value was None
                 display_data[pnl_key] = "N/A"


        # Filter out None values before printing for cleaner display, but keep "N/A" strings and formatted values
        # Create a new dictionary containing only non-None values.
        # Use .get() with None default for safety, then check explicitly for None.
        display_data_filtered = {
            k: v for k, v in display_data.items()
            if v is not None
        }

        # Check if essential fields are missing after filtering (excluding optional V5 fields)
        essential_keys = ["Total Equity", "Available Balance", "Used Margin"]
        # Check if the *values* for essential keys are None or 'N/A' after formatting
        missing_essentials = [k for k in essential_keys if display_data_filtered.get(k) in (None, 'N/A')]

        if not display_data_filtered or missing_essentials:
             self.ui.print_warning("Balance data received but key fields (Equity, Available, Used) are missing, null, or unparsable.")
             if missing_essentials: self.ui.print_warning(f"Missing essential fields: {', '.join(missing_essentials)}")
             self.ui.print_info("Raw balance data received:")
             logger.info(f"Raw balance structure causing issues: {balance_data}")
             # Use print_table for the raw dict if possible, with high precision for potential raw numbers
             self.ui.print_table(balance_data, title="Raw Balance Data (Debug)", float_format='{:.8f}')
        else:
             # Let print_table handle the dictionary containing formatted numbers and colored strings
             # Use '{:.4f}' as a fallback format for any numeric values not explicitly formatted as strings
             self.ui.print_table(display_data_filtered, title="USDT Futures Account Balance", float_format='{:.4f}')

             self.ui.print_info("\nNote: PNL figures are account-wide estimates from balance data, if available.")
             self.ui.print_info("Use 'View Open Positions' for PNL per individual position.")

        await self.ui_wait_for_enter_async()

    async def view_open_orders(self):
        """Fetches and displays open orders, formatting key details."""
        self.ui.display_header("View Open Order Scrolls")
        # Allow empty input for 'all symbols' filter
        # Use async input helper with validation
        symbol_input = await self.ui_get_input_async(
            "Enter symbol to filter (e.g., BTC/USDT) or leave blank for all",
            required=False, # Input is optional
            # Validation: if input is NOT blank, validate it as a symbol. If blank, it's valid.
            validation_func=lambda s: True if not s or not str(s).strip() else self._validate_symbol(str(s))
            # Note: validation_func receives the *converted* type, which is str here as input_type=str by default
        )
        # Handle potential error propagation or cancellation from get_input
        if symbol_input is None and (symbol_input != "" and symbol_input is not False): # Check for None explicitly, allow empty string
             # If validation returned False (should return str error instead), handle that too.
             # If ui_get_input_async returned None because input was optional and left blank, proceed.
             # If it returned None because of interruption/error, it would raise an exception caught by handle_action.
             pass # Proceed if symbol_input is None (optional blank input) or empty string ""

        # Normalize symbol only if a non-empty string was provided
        symbol = self._get_ccxt_symbol(symbol_input) if symbol_input and isinstance(symbol_input, str) else None

        action_desc = f"for {symbol}" if symbol else "for all symbols"
        self.ui.print_info(f"{self.ui.colors['primary']}# Fetching active order scrolls {action_desc}...{self.ui.colors['reset']}")

        # Fetch open orders using client method (handles V5 params)
        # Pass the normalized symbol or None
        open_orders = await self.exchange_client.fetch_open_orders(symbol=symbol)

        if not open_orders:
            self.ui.print_warning(f"No open orders found {action_desc}.")
        else:
            # Select, rename, and format columns for clarity using standard CCXT keys
            display_data = []
            for order in open_orders:
                # Use .get() with defaults for robustness
                order_info = {
                    "ID": order.get('id', 'N/A'),
                    # Format datetime nicely, handle missing/invalid
                    "Timestamp": pd.to_datetime(order.get('datetime')).strftime('%Y-%m-%d %H:%M:%S UTC') if pd.notna(pd.to_datetime(order.get('datetime'), errors='coerce')) else 'N/A',
                    "Symbol": order.get('symbol', 'N/A'),
                    "Type": str(order.get('type', 'N/A')).capitalize(), # Capitalize type (limit, market etc)
                    "Side": str(order.get('side', 'N/A')).capitalize(), # Capitalize side (buy, sell)
                    "Price": order.get('price'), # Let print_table format float
                    "Amount": order.get('amount'),
                    "Filled": order.get('filled'),
                    "Remaining": order.get('remaining'),
                    "Status": str(order.get('status', 'N/A')).capitalize(), # Capitalize status
                    # Standard CCXT keys for triggers (might be None)
                    "SL Price": order.get('stopLossPrice'),
                    "TP Price": order.get('takeProfitPrice'),
                    # Consolidate Trail Info
                    "Trail Info": "N/A", # Placeholder
                    "ReduceOnly": order.get('reduceOnly', False), # Default to False if missing, will be formatted below
                }

                # Consolidate Trailing Stop information
                trail_info_str = "N/A"
                trail_percent = order.get('trailingPercent') # Standard CCXT key (float)
                trail_active_price = order.get('trailingActivationPrice') # Another possible standard key (float)
                # Check Bybit V5 specific info in the 'info' dict if standard keys are missing/uninformative
                bybit_trail_val = None
                if 'info' in order and isinstance(order['info'], dict):
                     bybit_trail_val = order['info'].get('trailingStop') # Bybit V5 key (string value like "0.5")

                if trail_percent is not None and trail_percent > 0:
                     # Format the percentage value
                     trail_info_str = f"{trail_percent:.2f}%"
                     # Add activation price if available (standard key)
                     if trail_active_price is not None: trail_info_str += f" (Act: {trail_active_price:.4f})"
                elif bybit_trail_val and bybit_trail_val != "0": # Check Bybit specific if standard missing and not "0"
                     trail_info_str = f"{bybit_trail_val}% (Info)" # Indicate source and show raw string
                     # Check for activation price in info as well (Bybit V5 key 'activePrice')
                     bybit_active_price = order['info'].get('activePrice')
                     if bybit_active_price: trail_info_str += f" (Act: {bybit_active_price})" # Show raw string


                order_info["Trail Info"] = trail_info_str

                # Add Trigger info from 'info' if standard keys are missing (redundancy check)
                # Use .get() with default None for safety
                if order_info["SL Price"] is None and order.get('info', {}).get('stopLoss'): order_info["SL Price"] = f"{order['info']['stopLoss']} (Info)"
                if order_info["TP Price"] is None and order.get('info', {}).get('takeProfit'): order_info["TP Price"] = f"{order['info']['takeProfit']} (Info)"

                # Format boolean ReduceOnly as Yes/No string
                order_info["ReduceOnly"] = "Yes" if order_info.get("ReduceOnly") else "No"


                # Filter out columns that are None (but keep 'N/A' strings) before appending
                # This makes the table cleaner if many optional fields are empty
                filtered_order_info = {k: v for k, v in order_info.items() if v is not None}

                display_data.append(filtered_order_info)

            # Define column order for the table
            # Include all potential keys, print_table will only show columns present in the data
            column_order = [
                "ID", "Timestamp", "Symbol", "Type", "Side", "Price", "Amount",
                "Filled", "Remaining", "Status", "SL Price", "TP Price", "Trail Info",
                "ReduceOnly", "ClientOrderId", "Cost", "Average", "Fee", "SL Price (Info)",
                "TP Price (Info)", "Trail Info (Info)"
            ]
            # Filter display_data to only include keys in column_order AND present in data, maintain order
            final_display_data = []
            # Get the set of all keys present in the data to know which columns exist
            all_present_keys = set()
            for row_dict in display_data:
                 all_present_keys.update(row_dict.keys())

            # Determine the final list of columns to display, maintaining order where possible
            final_columns_ordered = [col for col in column_order if col in all_present_keys]
            # Add any keys present in the data but not in the predefined order (e.g., unexpected fields)
            other_keys = sorted(list(all_present_keys - set(column_order)))
            final_columns_ordered.extend(other_keys)


            # Reconstruct display_data to ensure columns are in the desired order
            ordered_display_data = []
            for row_dict in display_data:
                 ordered_row = {col: row_dict.get(col) for col in final_columns_ordered}
                 ordered_display_data.append(ordered_row)


            # Use higher precision for price/amount/filled/remaining/average/cost/fee in crypto
            # print_table float_format applies to all numeric columns unless they are strings
            # We already formatted PNL and percentages as strings with colors, which is good.
            # Let's use '{:.8f}' as the default for unformatted floats like Price, Amount, Filled, etc.
            self.ui.print_table(ordered_display_data, title=f"Open Orders {action_desc}", float_format='{:.8f}')

        await self.ui_wait_for_enter_async()

    async def view_open_positions(self):
        """Fetches and displays open (non-zero size) positions with PNL and formatting."""
        self.ui.display_header("View Open Position Essences")
        # Allow empty input for 'all symbols' filter
        # Use async input helper with validation
        symbol_input = await self.ui_get_input_async(
            "Enter symbol to filter (e.g., BTC/USDT) or leave blank for all",
            required=False, # Input is optional
            validation_func=lambda s: True if not s or not str(s).strip() else self._validate_symbol(str(s))
        )
        # Handle potential error propagation or cancellation from get_input
        if symbol_input is None and (symbol_input != "" and symbol_input is not False):
             pass # Proceed if symbol_input is None (optional blank input) or empty string ""

        # Normalize symbol only if a non-empty string was provided
        symbol = self._get_ccxt_symbol(symbol_input) if symbol_input and isinstance(symbol_input, str) else None

        action_desc = f"for {symbol}" if symbol else "for all symbols"
        self.ui.print_info(f"{self.ui.colors['primary']}# Fetching open position essences {action_desc}...{self.ui.colors['reset']}")

        # Fetch positions (client method already filters for non-zero size and handles V5 params)
        # Pass the normalized symbol or None
        positions = await self.exchange_client.fetch_positions(symbols=symbol)

        if not positions:
            self.ui.print_warning(f"No open positions found {action_desc}.")
        else:
            # Select, rename, calculate, and format columns using standard CCXT keys
            display_data = []
            total_unrealized_pnl = 0.0 # Track total uPNL for summary, only sum if numeric and not None

            for pos in positions:
                # Extract data using standard CCXT keys with .get() for safety
                symbol = pos.get('symbol', 'N/A')
                side = str(pos.get('side', 'N/A')).capitalize() # 'long' or 'short' -> 'Long' or 'Short'
                # Size extraction (already validated in fetch_positions, but re-extract for display)
                # Use .get() with None default, then try different keys
                size_raw = pos.get('contracts') # Standard CCXT field for size in base currency
                if size_raw is None: size_raw = pos.get('contractSize') # Fallback
                if size_raw is None and 'info' in pos and isinstance(pos.get('info'), dict):
                     size_raw = pos['info'].get('size') # Bybit V5 info size
                # Convert size to float if possible, default to 0.0 on failure/None
                size = float(size_raw) if size_raw is not None and size_raw != '' else 0.0

                # Get other raw values
                entry_price = pos.get('entryPrice')
                mark_price = pos.get('markPrice')
                liq_price = pos.get('liquidationPrice')
                initial_margin = pos.get('initialMargin')
                # maint_margin = pos.get('maintenanceMargin') # Less commonly displayed
                unrealized_pnl = pos.get('unrealizedPnl')
                leverage = pos.get('leverage')
                notional = pos.get('notional') # Position value (Size * Price)

                # Prepare dictionary for the table row
                pos_info = {
                    "Symbol": symbol,
                    "Side": side,
                    "Size": size, # Let print_table format float
                    "Entry Price": entry_price,
                    "Mark Price": mark_price,
                    "Liq Price": liq_price,
                    "Margin": initial_margin, # Initial margin used
                    "uPNL": unrealized_pnl, # Placeholder, formatted with color below
                    "uPNL %": "N/A", # Placeholder, calculated and colored below
                    "Leverage": f"{leverage}x" if leverage is not None else 'N/A', # Format leverage as string
                    "Notional": notional, # Value of position
                }

                # Calculate percentage PNL based on initial margin if possible
                pnl_percent_str = "N/A"
                # Ensure initial_margin is numeric and non-zero before calculating percentage
                initial_margin_float = float(initial_margin) if initial_margin is not None else 0.0
                unrealized_pnl_float = float(unrealized_pnl) if unrealized_pnl is not None else None

                if unrealized_pnl_float is not None and not np.isclose(initial_margin_float, 0.0):
                     try:
                          pnl_percent = (unrealized_pnl_float / initial_margin_float) * 100
                          pnl_color = self.ui.colors['positive'] if pnl_percent > 0 else self.ui.colors['negative'] if pnl_percent < 0 else self.ui.colors['neutral']
                          # Store the colored string for display, formatted to 2 decimal places
                          pnl_percent_str = f"{pnl_color}{pnl_percent:.2f}%{self.ui.colors['reset']}"
                     except (ValueError, TypeError, ZeroDivisionError) as calc_err:
                          logger.warning(f"Could not calculate PNL % for {symbol}: {calc_err}")
                          pnl_percent_str = f"{self.ui.colors['error']}Calc Error{self.ui.colors['reset']}" # Indicate calculation error

                pos_info["uPNL %"] = pnl_percent_str

                # Format uPNL value itself with color
                if unrealized_pnl_float is not None:
                    try:
                        # Use appropriate color for uPNL based on value
                        pnl_color = self.ui.colors['positive'] if unrealized_pnl_float > 0 else self.ui.colors['negative'] if unrealized_pnl_float < 0 else self.ui.colors['neutral']
                        # Override the uPNL value with the colored string, formatted to 4 decimal places
                        pos_info["uPNL"] = f"{pnl_color}{unrealized_pnl_float:.4f}{self.ui.colors['reset']}"
                        total_unrealized_pnl += unrealized_pnl_float # Add to total PNL sum
                    except (ValueError, TypeError) as parse_err:
                         logger.warning(f"Could not parse uPNL value '{unrealized_pnl}' for {symbol}: {parse_err}")
                         pos_info["uPNL"] = f"{self.ui.colors['error']}Parse Error{self.ui.colors['reset']}"
                else:
                     pos_info["uPNL"] = "N/A" # Ensure explicit N/A if missing

                # Filter out columns that are None (but keep 'N/A' strings and formatted values)
                filtered_pos_info = {k: v for k, v in pos_info.items() if v is not None}
                # Add the processed dict to the list for the table
                display_data.append(filtered_pos_info)

            # Define column order for the table
            # Include all potential keys, print_table will only show columns present in the data
            column_order = [
                "Symbol", "Side", "Size", "Entry Price", "Mark Price", "Liq Price",
                "Margin", "uPNL", "uPNL %", "Leverage", "Notional", "Maintenance Margin" # Add Maint Margin just in case
            ]
            # Filter display_data to only include keys in column_order AND present in data, maintain order
            final_display_data = []
            # Get the set of all keys present in the data to know which columns exist
            all_present_keys = set()
            for row_dict in display_data:
                 all_present_keys.update(row_dict.keys())

            # Determine the final list of columns to display, maintaining order where possible
            final_columns_ordered = [col for col in column_order if col in all_present_keys]
            # Add any keys present in the data but not in the predefined order (e.g., unexpected fields)
            other_keys = sorted(list(all_present_keys - set(column_order)))
            final_columns_ordered.extend(other_keys)

            # Reconstruct display_data to ensure columns are in the desired order
            ordered_display_data = []
            for row_dict in display_data:
                 ordered_row = {col: row_dict.get(col) for col in final_columns_ordered}
                 ordered_display_data.append(ordered_row)


            # Print table using the enhanced print_table which handles colored strings
            # Use appropriate float format (e.g., 4 decimals for price/margin, 8 for size if needed)
            # print_table float_format applies to unformatted numeric values
            self.ui.print_table(ordered_display_data, title=f"Open Positions {action_desc}", float_format='{:.4f}')

            # Print summary total PNL
            pnl_color = self.ui.colors['positive'] if total_unrealized_pnl > 0 else self.ui.colors['negative'] if total_unrealized_pnl < 0 else self.ui.colors['neutral']
            print(f"\n{self.ui.colors['info']}Total Unrealized PNL (Displayed Positions): {pnl_color}{total_unrealized_pnl:.4f}{self.ui.colors['reset']}")

        await self.ui_wait_for_enter_async()

    async def check_price_menu(self):
        """Fetches and displays the current ticker information for a symbol."""
        self.ui.display_header("Check Price Pulse")
        # Use default from config, validate format and existence
        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        # Use async input helper with validation
        symbol_input = await self.ui_get_input_async("Enter symbol (e.g., BTC/USDT or BTCUSDT)", default=default_symbol, validation_func=self._validate_symbol)
        if symbol_input is None: return # User cancelled or error propagated
        symbol = self._get_ccxt_symbol(symbol_input) # Normalize after validation

        self.ui.print_info(f"{self.ui.colors['primary']}# Fetching current pulse for {symbol}...{self.ui.colors['reset']}")
        # Fetch ticker using client method (handles V5 params)
        ticker = await self.exchange_client.fetch_ticker(symbol)

        # Display relevant ticker info using print_table for dict
        # Use standard CCXT ticker keys, format floats and handle None
        display_data = {
            "Symbol": ticker.get('symbol'),
            # Format datetime nicely, handle missing/invalid
            "Timestamp": pd.to_datetime(ticker.get('datetime')).strftime('%Y-%m-%d %H:%M:%S UTC') if pd.notna(pd.to_datetime(ticker.get('datetime'), errors='coerce')) else 'N/A',
            "Last Price": ticker.get('last'),
            "Mark Price": ticker.get('mark'), # Important for liquidation/PNL
            "Index Price": ticker.get('index'), # Often used for funding/settlement
            "Bid Price": ticker.get('bid'), # Highest price buyers willing to pay
            "Ask Price": ticker.get('ask'), # Lowest price sellers willing to accept
            "Spread": None, # Calculated below, will be added only if not None
            "High (24h)": ticker.get('high'),
            "Low (24h)": ticker.get('low'),
            "Volume (24h Base)": ticker.get('baseVolume'), # e.g., BTC volume
            "Volume (24h Quote)": ticker.get('quoteVolume'), # e.g., USDT volume
            "Change (24h)": ticker.get('change'), # Absolute change
            "% Change (24h)": ticker.get('percentage'), # Standard CCXT key, will be formatted below
            "Funding Rate (Est)": ticker.get('fundingRate'), # Standard CCXT key, will be formatted below
            # Format next funding time, handle missing/invalid timestamp
            "Next Funding Time": pd.to_datetime(ticker.get('fundingTimestamp'), unit='ms').strftime('%Y-%m-%d %H:%M:%S UTC') if pd.notna(pd.to_datetime(ticker.get('fundingTimestamp'), unit='ms', errors='coerce')) else None, # Format time only
        }

        # Calculate Spread
        bid = display_data.get('Bid Price') # Use values already extracted
        ask = display_data.get('Ask Price')
        if bid is not None and ask is not None:
            try:
                bid_float = float(bid)
                ask_float = float(ask)
                # Check if conversion was successful and numbers are finite
                if np.isfinite(bid_float) and np.isfinite(ask_float):
                    display_data["Spread"] = abs(ask_float - bid_float)
                else:
                     display_data["Spread"] = f"{self.ui.colors['warning']}Invalid Price Data{self.ui.colors['reset']}"
            except (ValueError, TypeError):
                 display_data["Spread"] = f"{self.ui.colors['error']}Calc Error{self.ui.colors['reset']}"


        # Colorize % Change and format
        perc_change = display_data.get("% Change (24h)")
        if perc_change is not None:
             try:
                 perc_float = float(perc_change)
                 perc_color = self.ui.colors['positive'] if perc_float > 0 else self.ui.colors['negative'] if perc_float < 0 else self.ui.colors['neutral']
                 display_data["% Change (24h)"] = f"{perc_color}{perc_float:.2f}%{self.ui.colors['reset']}"
             except (ValueError, TypeError):
                 display_data["% Change (24h)"] = f"{self.ui.colors['error']}Parse Error{self.ui.colors['reset']}"
        # Keep None if key was missing

        # Format Funding Rate if available
        funding_rate = display_data.get("Funding Rate (Est)")
        if funding_rate is not None:
             try:
                 # Funding rate is usually given as a rate (e.g., 0.0001), multiply by 100 for percentage
                 display_data["Funding Rate (Est)"] = f"{float(funding_rate) * 100:.4f}%"
             except (ValueError, TypeError):
                 display_data["Funding Rate (Est)"] = f"{self.ui.colors['error']}Parse Error{self.ui.colors['reset']}"
        # Keep None if key was missing

        # Filter out None values before printing for cleaner display, keep "N/A" and formatted strings
        display_data_filtered = {k: v for k, v in display_data.items() if v is not None}

        # Let print_table handle the dictionary with formatted numbers and colored strings
        # Use '{:.4f}' as a fallback format for any numeric values not explicitly formatted as strings
        self.ui.print_table(display_data_filtered, title=f"Current Ticker: {symbol}", float_format='{:.4f}')
        await self.ui_wait_for_enter_async()


    async def technical_analysis_menu(self):
        """Handles fetching data, calculating, and displaying technical analysis."""
        # Check for pandas_ta availability early
        if ta is None:
            self.ui.print_error("Technical Analysis requires the 'pandas_ta' library.")
            self.ui.print_warning("Please install it using: pip install pandas_ta")
            await self.ui_wait_for_enter_async()
            return

        self.ui.display_header("Technical Analysis Oracle")

        # --- Get Inputs ---
        # Use default from config, validate format and existence
        default_symbol = self.config_manager.config.get("default_symbol", "BTC/USDT")
        # Use async input helper with validation
        symbol_input = await self.ui_get_input_async("Enter symbol (e.g., BTC/USDT or BTCUSDT)", default=default_symbol, validation_func=self._validate_symbol)
        if symbol_input is None: return # User cancelled or error propagated
        symbol = self._get_ccxt_symbol(symbol_input) # Normalize after validation

        # Use default from config, validate format and exchange support (partial)
        default_timeframe = self.config_manager.config.get("default_timeframe", "1h")
        # Use async input helper with validation
        timeframe_input = await self.ui_get_input_async("Enter time-shard (e.g., 1m, 5m, 1h, 1d)", default=default_timeframe, validation_func=self._validate_timeframe)
        if timeframe_input is None: return
        # Normalize timeframe after validation passes (validator handles formats like 'D')
        timeframe = timeframe_input.strip().lower()
        tf_map = {'d': '1d', 'w': '1w', 'm': '1M'} # Map Bybit letters if used
        if timeframe.upper() in tf_map: timeframe = tf_map[timeframe.upper()]


        # --- Fetch Data ---
        # Determine data fetch limit based on chart points and indicator lookbacks
        chart_points = self.config_manager.config.get("chart_points", 60)
        indicator_periods = self.config_manager.config.get("indicator_periods", {})
        max_lookback_needed = 0
        try:
             # Calculate max lookback needed by configured indicators + buffer
             # Ensure periods are integers/floats before using in max()
             indicator_lengths = []
             for key, val in indicator_periods.items():
                 try:
                      # Attempt to convert to float first, then int if it's a whole number
                      float_val = float(val)
                      if np.isfinite(float_val) and float_val > 0: # Only consider positive finite numbers
                           if float_val.is_integer(): indicator_lengths.append(int(float_val))
                           else: indicator_lengths.append(float_val)
                 except (ValueError, TypeError):
                      logger.warning(f"Could not parse indicator period '{key}' value '{val}' as number.")

             # Add MACD default periods (12, 26 for EMAs, 9 for signal = ~35-40 lookback needed)
             indicator_lengths.extend([12, 26, 9])

             # Add Stoch calculation lengths (K, D, SmoothK) - pandas_ta uses cumulative effect
             # A simple sum k+d+smooth_k is a rough estimate for data points needed for the *last* value
             # A more accurate estimate is k + max(d, smooth_k) or similar, but k+d+smooth_k is safer.
             try:
                 stoch_k = int(indicator_periods.get('stoch_k', 14))
                 stoch_d = int(indicator_periods.get('stoch_d', 3))
                 stoch_smooth_k = int(indicator_periods.get('stoch_smooth_k', 3))
                 indicator_lengths.append(stoch_k + stoch_d + stoch_smooth_k)
             except (ValueError, TypeError):
                 logger.warning("Could not parse Stochastic periods for lookback estimation.")
                 indicator_lengths.append(14 + 3 + 3) # Use default stoch estimate


             if indicator_lengths:
                 max_lookback_needed = max(indicator_lengths)
             # Add a buffer (e.g., 50-100 candles) for indicator warmup/NaNs
             buffer = 75 # A reasonable buffer
             max_lookback_needed += buffer
        except Exception as e:
             logger.warning(f"{self.ui.colors['warning']}Could not calculate max indicator lookback from config: {e}. Using default 250.{self.ui.colors['reset']}")
             max_lookback_needed = 250 # Default large lookback if calculation fails

        # Fetch enough data: max(chart points, indicator lookback) + buffer
        # Respect potential exchange limits (e.g., Bybit max 1000 per request for V5)
        # Fetch slightly more than strictly needed to ensure enough data after cleaning/NaNs
        fetch_limit = min(max(chart_points + 50, max_lookback_needed), 1000) # Add small buffer for chart, cap at 1000

        self.ui.print_info(f"{self.ui.colors['primary']}# Scrying market data for {symbol} ({timeframe}, limit={fetch_limit})...{self.ui.colors['reset']}")
        # Fetch OHLCV data (handle_action will catch errors from fetch_ohlcv, including invalid timeframe)
        ohlcv_data = await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)

        if not ohlcv_data:
            self.ui.print_warning(f"No OHLCV data received for {symbol} ({timeframe}). Cannot perform analysis.")
            await self.ui_wait_for_enter_async()
            return

        # --- Process Data into DataFrame ---
        df: Optional[pd.DataFrame] = None # Initialize df
        try:
            # Create DataFrame, ensure columns are lowercase strings
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.columns = [str(col).lower() for col in df.columns] # Ensure lowercase

            # Convert timestamp to datetime and set as index (UTC assumed from CCXT)
            # Use errors='coerce' in case of bad timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            df.set_index('timestamp', inplace=True)

            # Convert OHLCV columns to numeric, coercing errors (handles None, strings, etc.)
            # Use errors='coerce' to turn non-numeric values into NaN
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

            # Log data points received *before* cleaning
            logger.info(f"Received {len(df)} data points from exchange for {symbol} ({timeframe}).")

            # Drop rows where essential data (close, high, low) is missing AFTER conversion
            initial_len = len(df)
            df.dropna(subset=['close', 'high', 'low'], inplace=True) # Need HLC for most indicators/pivots
            # Drop rows with NaN index (from timestamp coerce)
            df.dropna(axis=0, subset=[df.index.name], inplace=True)

            if len(df) < initial_len:
                 logger.warning(f"{self.ui.colors['warning']}Dropped {initial_len - len(df)} rows with NaN OHLC or timestamp data during processing.{self.ui.colors['reset']}")

            if df.empty:
                 # Raise error if no valid data remains after cleaning
                 raise ValueError("DataFrame is empty after removing NaN/invalid data.")

            logger.info(f"Processed {len(df)} valid data points after cleaning.")
            # Sort by timestamp just in case data isn't perfectly ordered (though fetch_ohlcv usually is)
            df.sort_index(inplace=True)

        except ValueError as ve: # Catch specific error from above
             logger.error(f"{self.ui.colors['error']}Error processing OHLCV data: {ve}{self.ui.colors['reset']}")
             self.ui.print_error(f"Failed to process market data: {ve}")
             await self.ui_wait_for_enter_async()
             return
        except Exception as e:
             logger.error(f"{self.ui.colors['error']}Unexpected error processing OHLCV data into DataFrame: {e}{self.ui.colors['reset']}", exc_info=True)
             self.ui.print_error(f"Failed to process market data: {e}")
             await self.ui_wait_for_enter_async()
             return

        # --- Calculate Indicators ---
        # Pass the cleaned, processed df to the calculator
        df_indicators = self.ta_analyzer.calculate_indicators(df)

        # Check if indicators were actually calculated (TA func logs details)
        # Compare DataFrames carefully, check if columns were added
        # If calculate_indicators failed (returned original df or an empty df), df_indicators will be df or empty
        if df_indicators is df or df_indicators.empty:
             # If df_indicators is the same object as df or is empty, it means TA failed or had no data
             # The TA func logs warnings/errors in this case.
             # We can still proceed to show pivots/chart if the original df is valid.
             self.ui.print_warning("Technical analysis calculation failed or produced no new indicators.")
             if df_indicators.empty:
                  logger.warning("TA calculation returned an empty DataFrame.")
                  # If TA returned empty, we cannot proceed with chart/detailed table either.
                  self.ui.print_error("TA calculation resulted in an empty DataFrame. Cannot display indicators or chart.")
                  await self.ui_wait_for_enter_async()
                  return
             # If TA failed but df is still valid, proceed using df for chart/pivots
             df_to_use_for_display = df # Use the original cleaned df
        else:
             # TA calculation was successful and returned a new DataFrame with indicators
             logger.info("Indicators calculated successfully.")
             df_to_use_for_display = df_indicators # Use the df with indicators


        # --- Calculate Pivot Points ---
        pivot_points: Optional[Dict] = None
        # Get pivot period from config (already normalized during load/save)
        pivot_period_str = self.config_manager.config.get("pivot_period", "1d")
        try:
            # Validate the pivot timeframe from config before fetching
            validation_result = self._validate_timeframe(pivot_period_str)
            if validation_result is not True:
                 # Log error and notify user, but don't halt the entire analysis
                 logger.error(f"{self.ui.colors['error']}Invalid pivot timeframe '{pivot_period_str}' in config: {validation_result}{self.ui.colors['reset']}")
                 self.ui.print_warning(f"Skipping Pivot Points: Invalid timeframe '{pivot_period_str}' in config.")
            else:
                self.ui.print_info(f"{self.ui.colors['primary']}# Fetching data for {pivot_period_str} pivot points...{self.ui.colors['reset']}")
                # Fetch last 2 candles of the pivot timeframe. Index 0 is the last *completed* candle. Index 1 is the current (incomplete).
                # Add small delay to avoid potential rate limits after main data fetch
                await asyncio.sleep(0.6) # Slightly more than 0.5s
                # Use same symbol, fetch 2 candles of the pivot timeframe
                pivot_ohlcv = await self.exchange_client.fetch_ohlcv(symbol, pivot_period_str, limit=2)

                if pivot_ohlcv and len(pivot_ohlcv) >= 1:
                    # Create DataFrame for the pivot period data
                    df_pivot_raw = pd.DataFrame(pivot_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_pivot_raw.columns = [str(col).lower() for col in df_pivot_raw.columns] # Ensure lowercase
                    # Convert timestamp and set index, handle errors
                    df_pivot_raw['timestamp'] = pd.to_datetime(df_pivot_raw['timestamp'], unit='ms', utc=True, errors='coerce')
                    df_pivot_raw.set_index('timestamp', inplace=True)
                    # Convert to numeric, coerce errors for HLC
                    for col in ['high', 'low', 'close']: # Only need HLC for pivots
                         df_pivot_raw[col] = pd.to_numeric(df_pivot_raw[col], errors='coerce')
                    # Drop rows with NaN index or NaN HLC
                    df_pivot_raw.dropna(axis=0, subset=[df_pivot_raw.index.name, 'high', 'low', 'close'], inplace=True)

                    # Sort just in case
                    df_pivot_raw.sort_index(inplace=True)

                    # Use the first row provided (index 0), which should be the last *completed* period's HLC
                    if not df_pivot_raw.empty and len(df_pivot_raw) >= 1:
                         # Pass DataFrame slice containing only the row for calculation
                         pivot_calc_data = df_pivot_raw.iloc[[0]] # Pass as DataFrame slice
                         pivot_points = self.ta_analyzer.calculate_pivot_points(pivot_calc_data)
                         if pivot_points:
                             # Attempt to get timestamp from index for logging
                             candle_time = pivot_calc_data.index[0] if not pivot_calc_data.index.empty else "Unknown Time"
                             candle_time_str = candle_time.strftime('%Y-%m-%d %H:%M:%S %Z') if isinstance(candle_time, pd.Timestamp) else str(candle_time)
                             logger.info(f"Calculated {pivot_period_str} pivots based on candle ending {candle_time_str}")
                         else:
                             # Calc function already logged NaN/other issues
                             logger.warning(f"{self.ui.colors['warning']}Pivot calculation returned None for {pivot_period_str}. Check HLC data for the period.{self.ui.colors['reset']}")
                    else:
                         logger.warning(f"{self.ui.colors['warning']}Could not use fetched data for {pivot_period_str} pivots after cleaning (e.g., NaN HLC or empty df).{self.ui.colors['reset']}")
                else:
                     logger.warning(f"{self.ui.colors['warning']}Could not fetch sufficient data (need >= 1 completed candle) for {pivot_period_str} pivot points.{self.ui.colors['reset']}")

        # Catch errors during pivot fetch/calculation but don't stop the whole TA display
        except (ValueError, TypeError) as e: # Catch specific errors like invalid timeframe or data type issues
            logger.error(f"{self.ui.colors['error']}Error calculating pivot points ({pivot_period_str}): {e}{self.ui.colors['reset']}")
            self.ui.print_warning(f"Could not calculate pivot points ({pivot_period_str}): {e}")
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
             logger.error(f"{self.ui.colors['error']}CCXT Error fetching pivot data ({pivot_period_str}): {e}{self.ui.colors['reset']}")
             self.ui.print_warning(f"Could not fetch data for pivot points ({pivot_period_str}): {e}")
        except Exception as e: # Catch unexpected errors
            logger.error(f"{self.ui.colors['error']}Unexpected error fetching/calculating pivot points: {e}{self.ui.colors['reset']}", exc_info=True)
            self.ui.print_warning(f"Could not calculate pivot points ({pivot_period_str}): Unexpected error.")

        # --- Display Results ---

        # 1. TA Summary (uses df_indicators if available, handles colors)
        # Pass the DataFrame that contains indicator results if TA was successful, otherwise pass the original cleaned df
        ta_summary = self.ta_analyzer.generate_ta_summary(df_indicators if df_indicators is not df and not df_indicators.empty else df)
        self.ui.print_table(ta_summary, title=f"TA Summary: {symbol} ({timeframe})") # Let print_table handle dict/colors

        # 2. Pivot Points Table
        if pivot_points:
            self.ui.print_table(pivot_points, title=f"Classic Pivot Points (Based on last completed {pivot_period_str})", float_format='{:.4f}')
        else:
            self.ui.print_warning(f"Pivot points ({pivot_period_str}) could not be calculated or data unavailable.")

        # 3. Price Chart (using close prices from df_to_use_for_display)
        if df_to_use_for_display is not None and 'close' in df_to_use_for_display.columns:
            # Get close prices, drop NaNs, convert to list for asciichart
            # Use .copy() before dropna to avoid SettingWithCopyWarning if df_to_use_for_display is a slice/view
            close_prices = df_to_use_for_display['close'].dropna().tolist()
            if close_prices:
                # Use configured number of points or fewer if not enough data
                points_to_chart = min(len(close_prices), chart_points)
                # Log if fewer points are charted than requested
                if points_to_chart < chart_points:
                     self.ui.print_warning(f"Insufficient valid data points ({len(close_prices)}) for requested chart length ({chart_points}). Charting last {points_to_chart}.")

                # Display the chart using the last 'points_to_chart' data points
                self.ui.display_chart(close_prices[-points_to_chart:], f"{symbol} ({timeframe}) Close Price Trend (Last {points_to_chart} points)")
            else:
                 self.ui.print_warning("No valid close price data available for chart after cleaning.")
        elif df_to_use_for_display is not None:
             self.ui.print_warning("Close price column not found in data, cannot display chart.")
        else:
             # This case should ideally not happen if df was created successfully
             self.ui.print_warning("Dataframe is None after processing. Cannot display chart.")


        # 4. Detailed Indicator Table (Last N periods from df_indicators)
        # Only show this table if indicators were successfully calculated (df_indicators is not the original df and is not empty)
        if df_indicators is not df and not df_indicators.empty:
            num_detail_rows = 15 # Number of recent rows to show (increased slightly)
            if len(df_indicators) >= 1: # Check if there's at least one row
                rows_to_show = min(len(df_indicators), num_detail_rows)
                # Take tail, copy to avoid SettingWithCopyWarning, format index, make index a column
                display_df_tail = df_indicators.tail(rows_to_show).copy()

                # Format timestamp index nicely for display
                if isinstance(display_df_tail.index, pd.DatetimeIndex):
                     # Use a more readable format, including timezone if available
                     display_df_tail.index = display_df_tail.index.strftime('%y-%m-%d %H:%M %Z')
                # Reset index to make timestamp a regular column for tabulate/pandas print
                display_df_tail = display_df_tail.reset_index()
                # Rename the index column to something descriptive like 'Timestamp'
                display_df_tail.rename(columns={display_df_tail.columns[0]: 'Timestamp'}, inplace=True)


                # Select and reorder columns for display (use lowercase names from TA calculation)
                base_cols = ['Timestamp', 'open', 'high', 'low', 'close', 'volume']
                # Get indicator columns dynamically (those not in base_cols)
                # Ensure column names are strings before checking
                indicator_cols = sorted([col for col in display_df_tail.columns if col not in base_cols and isinstance(col, str)])
                # Ensure columns exist before selecting
                final_cols = [col for col in base_cols + indicator_cols if col in display_df_tail.columns]

                # Use print_table with the prepared DataFrame slice
                # Use higher precision for indicator values
                self.ui.print_table(display_df_tail[final_cols],
                                    title=f"Recent Data & Indicators (Last {rows_to_show})",
                                    float_format='{:.5f}', # Use 5 decimals for indicators and prices
                                    index=False) # Index is now the 'Timestamp' column
            else:
                 # This case should ideally be caught earlier if df_indicators is empty
                 self.ui.print_warning("No data available in indicator DataFrame for detailed table.")
        elif df_indicators is df:
             # Only print this if TA wasn't calculated but user might expect the detailed table
             self.ui.print_info("Skipping detailed indicator table as indicators were not calculated.")


        await self.ui_wait_for_enter_async() # Pause after showing all TA info


    async def settings_menu(self):
        """Handles the settings configuration menu using async input helpers."""
        settings_running = True
        # Keep running the settings menu loop as long as the settings loop is active AND the main app is running
        while settings_running and self._running: # Check global running flag too
            # Check shutdown flag at the start of the settings loop iteration
            if self._shutdown_event.is_set():
                logger.debug("Shutdown detected in settings menu, breaking loop.")
                break # Exit settings if shutdown is triggered

            # Reload current config state and colors for display each loop iteration
            # This ensures changes made within settings (like theme/log level) are reflected immediately
            current_config = self.config_manager.config
            colors = self.ui.colors # Use current UI colors

            # Build options dynamically with current values shown
            options_map = {
                "1": ("Theme", current_config.get('theme')),
                "2": ("Default Symbol", current_config.get('default_symbol')),
                "3": ("Default Timeframe", current_config.get('default_timeframe')),
                "4": ("Default Order Type", current_config.get('default_order_type')),
                "5": ("Log Level", current_config.get('log_level')),
                "6": ("Pivot Period", current_config.get('pivot_period')),
                "7": ("Chart Points", current_config.get('chart_points')),
                "8": ("Chart Height", current_config.get('chart_height')),
                # TODO: Add indicator period settings here if desired
                "9": ("Back to Main Menu", None) # No current value needed for exit option
            }
            # Format options for display using current theme colors
            options_display = [
                # Use accent color for the current value part, ensure menu_option color for text
                f"{v[0]} [{colors['accent']}{v[1]}{colors['menu_option']}]" if v[1] is not None else v[0]
                for k, v in options_map.items()
            ]

            # Use async helper for menu display (runs blocking UI in executor)
            # Wrap the executor call in a helper coroutine for clarity
            async def _get_settings_choice():
                 loop = asyncio.get_running_loop()
                 # Use partial to pass arguments to the blocking function
                 func_call = partial(self.ui.display_menu, "Settings Configuration", options_display, "Select setting to change or Back")
                 try:
                     # Run the blocking function in executor
                     return await loop.run_in_executor(None, func_call)
                 except (EOFError, KeyboardInterrupt) as e:
                     raise e # Re-raise interrupts to be caught below
                 except Exception as e:
                     # Wrap other exceptions from the executor/UI function
                     raise RuntimeError(f"Failed to display settings menu: {e}") from e

            # --- Get User Choice ---
            choice: Optional[str] = None
            try:
                 # Await the async helper to get the user's choice
                 choice = await _get_settings_choice()
                 # If _get_settings_choice returns None (e.g., optional blank input, not applicable here),
                 # or if an exception was caught and handled internally without re-raising,
                 # choice might be None. In this specific menu context, display_menu should always return a string choice or raise.
                 if choice is None:
                     logger.warning("Settings menu choice returned None unexpectedly. Continuing loop.")
                     continue # Retry menu display

            # Catch interruption signals propagated from the async input helper
            except (EOFError, KeyboardInterrupt) as e:
                 logger.info(f"Settings menu interrupted ('{type(e).__name__}'), returning to main menu.")
                 # Don't trigger full shutdown here, just exit the settings loop gracefully
                 settings_running = False # Signal the inner while loop to stop
                 # Consider saving pending changes? For now, just break.
                 break # Break inner while loop, return to main menu loop

            # Catch any other unexpected runtime errors from the _get_settings_choice helper itself
            except RuntimeError as e:
                 self.ui.print_error(str(e)) # The helper wraps the original exception message
                 await asyncio.sleep(1) # Pause before retrying
                 continue # Retry settings menu display


            # Check global flags again after input returns, before processing choice
            # This is important if shutdown was signalled *after* the user made a menu choice
            # but *before* the choice was processed.
            if not self._running or self._shutdown_event.is_set():
                logger.debug("Shutdown detected after settings input, breaking loop.")
                break # Break inner while loop

            # --- Handle Settings Choices ---
            needs_save = False # Flag to track if config needs saving

            try:
                if choice == "1": # Change Theme
                    # Use async input helper with validation
                    theme_input = await self.ui_get_input_async("Enter theme (dark/light)", default=current_config.get('theme'), validation_func=lambda t: str(t).strip().lower() in ['dark', 'light'] or "Must be 'dark' or 'light'")
                    if theme_input is not None: # Check if input was provided (not None)
                        new_theme = str(theme_input).strip().lower() # Ensure string and lowercase
                        if new_theme != current_config.get('theme'):
                            self.config_manager.config["theme"] = new_theme
                            # Update colors immediately for UI feedback in the next loop iteration
                            self.config_manager.theme_colors = self.config_manager._setup_theme_colors()
                            self.ui.colors = self.config_manager.theme_colors # Update UI instance's colors
                            needs_save = True
                            self.ui.print_success(f"Theme changed to {new_theme}. Applied immediately.")
                        else:
                            self.ui.print_info("Theme unchanged.")
                    # Short pause for user to see message before menu redraws
                    await asyncio.sleep(0.8)

                elif choice == "2": # Set Default Symbol
                    # Use async input helper with validation, including market check
                    symbol_input = await self.ui_get_input_async("Enter default symbol (e.g., BTC/USDT)", default=current_config.get('default_symbol'), validation_func=self._validate_symbol)
                    if symbol_input is not None:
                        # Normalize the input symbol first using the helper
                        normalized_symbol = self._get_ccxt_symbol(str(symbol_input))
                        # Re-validate the *normalized* symbol before saving to ensure it's valid for CCXT/Bybit
                        validation_result = self._validate_symbol(normalized_symbol)
                        if validation_result is True: # Validation passed
                            if normalized_symbol != current_config.get('default_symbol'):
                                self.config_manager.config["default_symbol"] = normalized_symbol
                                needs_save = True
                                self.ui.print_success(f"Default symbol set to {normalized_symbol}.")
                            else:
                                self.ui.print_info("Default symbol unchanged.")
                        else:
                             # Validation failed, show the error message from validator
                             self.ui.print_error(f"Validation failed for symbol '{normalized_symbol}': {validation_result}")
                    await asyncio.sleep(0.8)

                elif choice == "3": # Set Default Timeframe
                     # Use async input helper with validation
                     tf_input = await self.ui_get_input_async("Enter default time-shard (e.g., 1h, 1d, W)", default=current_config.get('default_timeframe'), validation_func=self._validate_timeframe)
                     if tf_input is not None:
                         # Normalize timeframe (lowercase, map D/W/M)
                         new_tf_norm = str(tf_input).strip().lower()
                         tf_map = {'d': '1d', 'w': '1w', 'm': '1M'}
                         if new_tf_norm.upper() in tf_map: new_tf_norm = tf_map[new_tf_norm.upper()]

                         # Although _validate_timeframe does basic format check,
                         # exchange support is checked during fetch_ohlcv.
                         # We save it if the format is valid.
                         if new_tf_norm != current_config.get('default_timeframe'):
                             self.config_manager.config["default_timeframe"] = new_tf_norm
                             needs_save = True
                             self.ui.print_success(f"Default timeframe set to {new_tf_norm}.")
                         else:
                             self.ui.print_info("Default timeframe unchanged.")
                     await asyncio.sleep(0.8)

                elif choice == "4": # Set Default Order Type
                     # Use async input helper with validation
                     type_input = await self.ui_get_input_async("Enter default order type (Market/Limit)", default=current_config.get('default_order_type'), validation_func=self._validate_order_type)
                     if type_input is not None:
                         # Store capitalized version in config for consistency
                         new_type_cap = str(type_input).strip().capitalize()
                         if new_type_cap != current_config.get('default_order_type'):
                             self.config_manager.config["default_order_type"] = new_type_cap
                             needs_save = True
                             self.ui.print_success(f"Default order type set to {new_type_cap}.")
                         else:
                             self.ui.print_info("Default order type unchanged.")
                     await asyncio.sleep(0.8)

                elif choice == "5": # Set Log Level
                     log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                     # Use async input helper with validation
                     level_input = await self.ui_get_input_async(f"Enter log level ({'/'.join(log_levels)})", default=current_config.get('log_level'), validation_func=lambda L: str(L).strip().upper() in log_levels or f"Must be one of: {', '.join(log_levels)}")
                     if level_input is not None:
                         new_level_upper = str(level_input).strip().upper()
                         if new_level_upper != current_config.get('log_level'):
                             self.config_manager.config["log_level"] = new_level_upper
                             self.config_manager._apply_log_level() # Apply immediately
                             needs_save = True
                             self.ui.print_success(f"Log level set to {new_level_upper}. Applied immediately.")
                             # Log the change itself at the new level (if >= INFO)
                             logger.info(f"Log level changed to {new_level_upper} via settings.")
                         else:
                              self.ui.print_info("Log level unchanged.")
                     await asyncio.sleep(0.8)

                elif choice == "6": # Set Pivot Period
                     # Use async input helper with validation
                     pivot_tf_input = await self.ui_get_input_async("Enter timeframe for pivot points (e.g., 1h, 1d, 1W)", default=current_config.get('pivot_period'), validation_func=self._validate_timeframe)
                     if pivot_tf_input is not None:
                         # Normalize timeframe
                         new_pivot_tf_norm = str(pivot_tf_input).strip().lower()
                         tf_map = {'d': '1d', 'w': '1w', 'm': '1M'}
                         if new_pivot_tf_norm.upper() in tf_map: new_pivot_tf_norm = tf_map[new_pivot_tf_norm.upper()]

                         # Save the normalized pivot period if format is valid
                         if new_pivot_tf_norm != current_config.get('pivot_period'):
                             self.config_manager.config["pivot_period"] = new_pivot_tf_norm
                             needs_save = True
                             self.ui.print_success(f"Pivot point calculation period set to {new_pivot_tf_norm}.")
                         else:
                              self.ui.print_info("Pivot period unchanged.")
                     await asyncio.sleep(0.8)

                elif choice == "7": # Set Chart Points
                    # Use async input helper with integer range validation
                    min_points, max_points = 10, 1000
                    points_validator = lambda x: self._validate_integer_range(x, min_points, max_points)
                    points_input = await self.ui_get_input_async(f"Enter number of points for price chart ({min_points}-{max_points})", default=current_config.get('chart_points'), input_type=int, validation_func=points_validator)
                    if points_input is not None: # Input is guaranteed to be int by validation if not None
                         if points_input != current_config.get('chart_points'):
                             self.config_manager.config["chart_points"] = points_input
                             needs_save = True
                             self.ui.print_success(f"Chart points set to {points_input}.")
                         else:
                              self.ui.print_info("Chart points unchanged.")
                    await asyncio.sleep(0.8)

                elif choice == "8": # Set Chart Height
                    # Use async input helper with integer range validation
                    min_height, max_height = 5, 50
                    height_validator = lambda x: self._validate_integer_range(x, min_height, max_height)
                    height_input = await self.ui_get_input_async(f"Enter height for ASCII chart (lines, {min_height}-{max_height})", default=current_config.get('chart_height'), input_type=int, validation_func=height_validator)
                    if height_input is not None: # Guaranteed int if not None
                         if height_input != current_config.get('chart_height'):
                             self.config_manager.config["chart_height"] = height_input
                             needs_save = True
                             self.ui.print_success(f"Chart height set to {height_input}.")
                         else:
                              self.ui.print_info("Chart height unchanged.")
                    await asyncio.sleep(0.8)

                elif choice == "9": # Back to Main Menu
                    settings_running = False # Signal loop to stop
                    # Save config if changes were made before exiting settings
                    if needs_save:
                        self.config_manager.save_config() # save_config handles normalization
                        self.ui.print_info("Configuration saved.")
                        await asyncio.sleep(0.8) # Pause briefly before returning
                    # Loop will terminate, returning to main run loop

                else:
                    # Should not happen due to menu validation in ui.display_menu, but handle defensively
                    self.ui.print_error("Invalid settings choice received.")
                    await asyncio.sleep(1)

                # Save config after each successful change (or just before exiting on 'Back')
                # This provides more immediate persistence.
                if needs_save and settings_running: # Save if changed AND not exiting yet
                     self.config_manager.save_config() # save_config handles normalization
                     needs_save = False # Reset flag after saving

            # Handle interruptions during input *within* the settings loop
            # This is caught by the ui_get_input_async helper and re-raised
            except (EOFError, KeyboardInterrupt) as e:
                 logger.warning(f"Interruption '{type(e).__name__}' during settings modification. Exiting settings menu.")
                 settings_running = False # Exit settings loop
                 # Attempt to save pending changes on interrupt if any were made
                 if needs_save:
                      self.ui.print_warning("Attempting to save pending changes before exiting settings...")
                      self.config_manager.save_config() # Attempt save on interrupt if needed
                 # Do NOT trigger full shutdown here, just exit the settings menu
                 break # Break inner while loop immediately

            # Catch other errors propagated from the async input helper (e.g., RuntimeError from executor)
            except RuntimeError as e:
                 self.ui.print_error(f"Error during settings input: {e}")
                 logger.error(f"RuntimeError during settings input: {e}", exc_info=True)
                 # Attempt save on error too if changes were made before error
                 if needs_save:
                      self.ui.print_warning("Attempting to save any pending changes before continuing...")
                      self.config_manager.save_config()
                 await self.ui_wait_for_enter_async() # Pause before retrying settings menu

            # Catch any other unexpected errors during setting update/validation logic
            except Exception as e:
                 logger.error(f"Error handling setting choice '{choice}': {e}{self.ui.colors['reset']}", exc_info=True)
                 self.ui.print_error(f"An unexpected error occurred while changing setting: {e}")
                 if needs_save: # Attempt save on error too if changes were made before error
                      self.ui.print_warning("Attempting to save any pending changes before continuing...")
                      self.config_manager.save_config()
                 await self.ui_wait_for_enter_async() # Pause before retrying settings menu


# --- Main Execution ---

def check_create_env_file() -> bool:
    """Checks for .env file, creates a default one if missing, and checks for placeholders."""
    env_path = Path('.env')
    created_now = False
    print(f"{Fore.CYAN}Checking for API credential file '{env_path}'...{Style.RESET_ALL}")

    if not env_path.exists():
        print(f"{Fore.YELLOW}File '{env_path}' not found. Creating a default template...{Style.RESET_ALL}")
        try:
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write("# Bybit API Credentials (replace with your actual keys)\n")
                f.write("# Get keys from Bybit website: Account -> API Management\n")
                f.write
