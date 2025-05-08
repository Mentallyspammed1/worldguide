import asyncio
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Final, Type
import functools

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
)

# --- Constants ---
CONFIG_FILE_DEFAULT: Final[Path] = Path("config.yaml")
LOG_FILE: Final[Path] = Path("scalping_bot.log")
API_MAX_RETRIES: Final[int] = 5
API_RETRY_DELAY: Final[float] = 2.0  # seconds (increased default)
MIN_HISTORICAL_DATA_BUFFER: Final[int] = 10
MAX_ORDER_BOOK_DEPTH: Final[int] = 1000  # Practical limit for most exchanges

# --- Enums ---
class OrderType(str, Enum):
    """Enumeration for order types."""
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(str, Enum):
    """Enumeration for order sides."""
    BUY = "buy"
    SELL = "sell"

# --- Configuration Schema (Example - Replace with actual schema) ---
# Define a basic structure for validation; consider using Pydantic for robust validation
CONFIG_SCHEMA: Final[Dict[str, Type]] = {
    "exchange": str,
    "api_key": str,
    "api_secret": str,
    "symbol": str,
    "trade_amount_base": float,
    "profit_target_pct": float,
    "stop_loss_pct": float,
    "trade_loop_delay_sec": float,
    "simulation_mode": bool,
    "log_level": str,
    # Add other necessary configuration parameters and their expected types
}

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Setup Logger ---
def setup_logger(
    log_level: str = "INFO", log_file: Path = LOG_FILE
) -> logging.Logger:
    """Configures and returns a logger instance."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logger
    logger = logging.getLogger("ScalpingBot")
    logger.setLevel(numeric_level)

    # Prevent multiple handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        logger.error(f"Could not set up file logging to {log_file}: {e}")

    return logger

# Initialize logger (will be configured later based on config)
logger = setup_logger()

# --- API Retry Decorator (using Tenacity) ---
# Define specific exceptions to retry on
RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.RateLimitExceeded,
)

api_retry_strategy = retry(
    stop=stop_after_attempt(API_MAX_RETRIES),
    wait=wait_fixed(API_RETRY_DELAY),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,  # Reraise the exception if all retries fail
)

def retry_api_call(func):
    """Applies the tenacity retry strategy to an async function."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await api_retry_strategy(func)(*args, **kwargs)
    return wrapper


# --- Configuration Management ---
def validate_config(config: Dict[str, Any], schema: Dict[str, Type]) -> Dict[str, Any]:
    """
    Validates the loaded configuration against a basic schema.
    Raises ValueError if validation fails.
    Consider using Pydantic for more complex validation.
    """
    validated_config = {}
    errors = []
    for key, expected_type in schema.items():
        if key not in config:
            errors.append(f"Missing configuration key: '{key}'")
            continue
        value = config[key]
        if not isinstance(value, expected_type):
            errors.append(
                f"Invalid type for key '{key}': Expected {expected_type.__name__}, got {type(value).__name__}"
            )
        else:
            validated_config[key] = value

    # Check for unexpected keys (optional, but good practice)
    # for key in config:
    #     if key not in schema:
    #         logger.warning(f"Unexpected configuration key found: '{key}'")
    #         validated_config[key] = config[key] # Include unexpected keys if desired

    if errors:
        raise ValueError(f"Configuration validation failed:
" + "
".join(errors))

    return validated_config

def load_config(config_file: Path = CONFIG_FILE_DEFAULT) -> Dict[str, Any]:
    """Loads, validates, and returns configuration from a YAML file."""
    logger.info(f"Loading configuration from: {config_file}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with config_file.open("r") as f:
            config_raw = yaml.safe_load(f)
        if not isinstance(config_raw, dict):
            raise ValueError("Configuration file content must be a dictionary.")
        config = validate_config(config_raw, CONFIG_SCHEMA)
        logger.info("Configuration loaded and validated successfully.")
        return config
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing YAML configuration file {config_file}: {e}")
        raise
    except ValueError as e:
        logger.critical(f"Configuration validation error: {e}")
        raise
    except IOError as e:
        logger.critical(f"Error reading configuration file {config_file}: {e}")
        raise


# --- Scalping Bot Class ---
class ScalpingBot:
    """
    A cryptocurrency scalping bot using ccxt.

    Attributes:
        config (Dict[str, Any]): Bot configuration.
        exchange (ccxt_async.Exchange): Asynchronous ccxt exchange instance.
        symbol (str): Trading symbol (e.g., 'BTC/USDT').
        trade_amount_base (float): Amount of base currency to trade.
        profit_target_pct (float): Profit target percentage.
        stop_loss_pct (float): Stop loss percentage.
        trade_loop_delay (float): Delay between trading loop iterations.
        simulation_mode (bool): If True, simulate trades instead of executing.
        iteration (int): Counter for trading loop iterations.
        current_position (Optional[Dict[str, Any]]): Info about the current open position.
        historical_data (pd.DataFrame): DataFrame to store historical market data.
    """

    def __init__(self, config_file: Path = CONFIG_FILE_DEFAULT):
        """Initializes the ScalpingBot."""
        try:
            self.config = load_config(config_file)
        except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
            logger.critical(f"Failed to initialize bot due to config error: {e}")
            raise  # Re-raise to be caught by the main block

        # Reconfigure logger based on loaded config
        global logger
        logger = setup_logger(self.config.get("log_level", "INFO"))

        self.symbol: str = self.config["symbol"]
        self.trade_amount_base: float = self.config["trade_amount_base"]
        self.profit_target_pct: float = self.config["profit_target_pct"]
        self.stop_loss_pct: float = self.config["stop_loss_pct"]
        self.trade_loop_delay: float = self.config["trade_loop_delay_sec"]
        self.simulation_mode: bool = self.config["simulation_mode"]

        self.iteration: int = 0
        self.current_position: Optional[Dict[str, Any]] = None # Example: {'side': OrderSide.BUY, 'entry_price': 50000, 'amount': 0.001, 'order_id': '123'}
        self.historical_data: pd.DataFrame = pd.DataFrame()

        self._init_exchange()

    def _init_exchange(self):
        """Initializes the ccxt exchange instance."""
        exchange_id = self.config["exchange"]
        exchange_class = getattr(ccxt_async, exchange_id, None)

        if not exchange_class:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        exchange_config = {
            "apiKey": self.config.get("api_key"),
            "secret": self.config.get("api_secret"),
            # Add other exchange-specific options if needed
            # 'options': {'defaultType': 'spot'}, # Example
            'enableRateLimit': True, # Let ccxt handle basic rate limiting
        }
        # Remove None values for cleaner initialization
        exchange_config = {k: v for k, v in exchange_config.items() if v is not None}

        self.exchange = exchange_class(exchange_config)
        logger.info(f"Initialized exchange: {exchange_id}")
        # Consider adding a check here to ensure API keys are valid if not in simulation mode
        # e.g., by fetching balance or a small piece of market data


    @retry_api_call
    async def _fetch_ticker(self) -> Dict[str, Any]:
        """Fetches the latest ticker information for the symbol."""
        logger.debug(f"Fetching ticker for {self.symbol}")
        ticker = await self.exchange.fetch_ticker(self.symbol)
        return ticker

    @retry_api_call
    async def _fetch_ohlcv(self, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Fetches historical OHLCV data."""
        logger.debug(f"Fetching OHLCV data for {self.symbol} ({timeframe}, limit={limit})")
        if not self.exchange.has['fetchOHLCV']:
            logger.warning(f"Exchange {self.exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame() # Return empty DataFrame

        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    @retry_api_call
    async def _fetch_order_book(self, limit: int = 25) -> Dict[str, List[List[float]]]:
        """Fetches the order book."""
        logger.debug(f"Fetching order book for {self.symbol} (limit={limit})")
        # Ensure limit is within acceptable bounds
        safe_limit = min(limit, MAX_ORDER_BOOK_DEPTH)
        order_book = await self.exchange.fetch_order_book(self.symbol, limit=safe_limit)
        return order_book

    @retry_api_call
    async def _place_order(self, side: OrderSide, order_type: OrderType, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Places an order on the exchange."""
        log_price = f" at price {price}" if price is not None else ""
        logger.info(
            f"{Fore.YELLOW}Placing {side.value.upper()} {order_type.value.upper()} order "
            f"for {amount} {self.symbol.split('/')[0]}{log_price}{Style.RESET_ALL}"
        )

        if self.simulation_mode:
            logger.warning("--- SIMULATION MODE: Order not placed on exchange ---")
            # Simulate order execution (e.g., immediate fill at current price for market orders)
            simulated_order = {
                'id': f'sim_{int(time.time() * 1000)}',
                'timestamp': int(time.time() * 1000),
                'datetime': pd.Timestamp.now(tz='UTC').isoformat(),
                'symbol': self.symbol,
                'type': order_type.value,
                'side': side.value,
                'amount': amount,
                'price': price if price else (await self._fetch_ticker())['last'], # Simulate fill price
                'filled': amount,
                'status': 'closed', # Simulate immediate fill
                'fee': {'cost': amount * (price if price else (await self._fetch_ticker())['last']) * 0.001, 'currency': self.symbol.split('/')[1]}, # Simulate fee
                'info': {'simulated': True}
            }
            return simulated_order

        try:
            if order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Price must be specified for limit orders.")
                order = await self.exchange.create_limit_order(self.symbol, side.value, amount, price)
            elif order_type == OrderType.MARKET:
                order = await self.exchange.create_market_order(self.symbol, side.value, amount)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            logger.info(f"{Fore.GREEN}Order placed successfully: ID {order.get('id')}{Style.RESET_ALL}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"{Fore.RED}Insufficient funds to place {side.value} order: {e}{Style.RESET_ALL}")
            raise # Re-raise to be handled in the main loop
        except ccxt.ExchangeError as e:
            logger.error(f"{Fore.RED}Exchange error placing order: {e}{Style.RESET_ALL}")
            raise # Re-raise
        except Exception as e:
            logger.exception(f"{Fore.RED}Unexpected error placing order: {e}{Style.RESET_ALL}")
            raise # Re-raise


    async def _update_historical_data(self):
        """Fetches new OHLCV data and appends it to the historical data buffer."""
        # Fetch only the most recent data needed
        new_data = await self._fetch_ohlcv(timeframe='1m', limit=5) # Fetch a few recent candles
        if not new_data.empty:
            # Combine and remove duplicates, keeping the latest entry
            self.historical_data = pd.concat([self.historical_data, new_data])
            self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='last')]
            self.historical_data.sort_index(inplace=True)
            # Limit buffer size (optional, depends on strategy needs)
            self.historical_data = self.historical_data.tail(MIN_HISTORICAL_DATA_BUFFER * 10) # Keep a larger buffer
            logger.debug(f"Historical data updated. Current size: {len(self.historical_data)}")


    def _calculate_indicators(self) -> Dict[str, Any]:
        """
        Calculates trading indicators based on historical data.
        Replace with actual indicator calculations.
        """
        indicators = {}
        if len(self.historical_data) >= MIN_HISTORICAL_DATA_BUFFER:
            # Example: Calculate simple moving average
            try:
                indicators['sma_short'] = self.historical_data['close'].rolling(window=5).mean().iloc[-1]
                indicators['sma_long'] = self.historical_data['close'].rolling(window=20).mean().iloc[-1]
                logger.debug(f"Calculated indicators: SMA_short={indicators.get('sma_short', 'N/A')}, SMA_long={indicators.get('sma_long', 'N/A')}")
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                # Handle cases where there might not be enough data yet after filtering etc.
                return {} # Return empty dict if calculation fails
        else:
            logger.warning(f"Not enough historical data to calculate indicators (need {MIN_HISTORICAL_DATA_BUFFER}, have {len(self.historical_data)})")

        return indicators

    async def _manage_open_position(self, current_price: float):
        """Checks and manages the currently open position (stop-loss/take-profit)."""
        if not self.current_position:
            return # No open position to manage

        entry_price = self.current_position['entry_price']
        position_side = self.current_position['side']
        amount = self.current_position['amount']
        close_position = False
        close_reason = ""

        if position_side == OrderSide.BUY:
            # Check Take Profit
            profit_price = entry_price * (1 + self.profit_target_pct / 100)
            if current_price >= profit_price:
                logger.info(f"{Fore.GREEN}Take Profit triggered at {current_price} (Target: {profit_price}){Style.RESET_ALL}")
                close_position = True
                close_reason = "Take Profit"
            # Check Stop Loss
            stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
            if current_price <= stop_loss_price:
                logger.info(f"{Fore.RED}Stop Loss triggered at {current_price} (Target: {stop_loss_price}){Style.RESET_ALL}")
                close_position = True
                close_reason = "Stop Loss"

        elif position_side == OrderSide.SELL: # Handle short positions if applicable
             # Check Take Profit
            profit_price = entry_price * (1 - self.profit_target_pct / 100)
            if current_price <= profit_price:
                logger.info(f"{Fore.GREEN}Take Profit triggered at {current_price} (Target: {profit_price}){Style.RESET_ALL}")
                close_position = True
                close_reason = "Take Profit"
            # Check Stop Loss
            stop_loss_price = entry_price * (1 + self.stop_loss_pct / 100)
            if current_price >= stop_loss_price:
                logger.info(f"{Fore.RED}Stop Loss triggered at {current_price} (Target: {stop_loss_price}){Style.RESET_ALL}")
                close_position = True
                close_reason = "Stop Loss"

        if close_position:
            logger.info(f"Closing {position_side.value} position due to: {close_reason}")
            close_side = OrderSide.SELL if position_side == OrderSide.BUY else OrderSide.BUY
            try:
                # Use market order to ensure closure
                order_result = await self._place_order(side=close_side, order_type=OrderType.MARKET, amount=amount)
                # Basic PnL calculation (ignoring fees for simplicity here)
                exit_price = order_result.get('price', current_price) # Use actual fill price if available
                pnl = (exit_price - entry_price) * amount if position_side == OrderSide.BUY else (entry_price - exit_price) * amount
                pnl_pct = ((exit_price / entry_price) - 1) * 100 if position_side == OrderSide.BUY else ((entry_price / exit_price) - 1) * 100
                logger.info(f"{Fore.CYAN}Position closed. Entry: {entry_price}, Exit: {exit_price}, PnL: {pnl:.4f} {self.symbol.split('/')[1]} ({pnl_pct:.2f}%){Style.RESET_ALL}")
                self.current_position = None # Clear position after closing
            except Exception as e:
                # Error handling for closing order is crucial
                logger.error(f"Failed to close position automatically: {e}. Manual intervention might be required.")
                # Decide on retry logic or pausing here based on error type


    def _generate_trade_signal(self, indicators: Dict[str, Any], current_price: float) -> Optional[OrderSide]:
        """
        Generates a trade signal based on indicators and current price.
        Replace with actual trading logic.
        Returns OrderSide.BUY, OrderSide.SELL, or None.
        """
        if self.current_position:
            return None # Don't generate new signals if already in a position

        sma_short = indicators.get('sma_short')
        sma_long = indicators.get('sma_long')

        if sma_short is None or sma_long is None:
            logger.debug("Indicators not available, no signal generated.")
            return None

        # Example Crossover Strategy
        if sma_short > sma_long:
            # Potential buy signal (add more conditions)
            logger.debug(f"Potential BUY signal: SMA_short ({sma_short:.2f}) > SMA_long ({sma_long:.2f})")
            # Add checks: ensure not already bought, check volume, RSI, etc.
            # For this example, we'll just return BUY if condition met
            return OrderSide.BUY
        elif sma_short < sma_long:
            # Potential sell signal (add more conditions)
            logger.debug(f"Potential SELL signal: SMA_short ({sma_short:.2f}) < SMA_long ({sma_long:.2f})")
            # Add checks: ensure not already shorted, check volume, RSI, etc.
            # For this example, we'll just return SELL if condition met
            # Note: Simple scalping might only focus on long entries
            # return OrderSide.SELL # Uncomment if shorting is part of the strategy
            return None # Only long for this example
        else:
            logger.debug("No clear signal based on SMA crossover.")
            return None


    async def _execute_trade_signal(self, signal: OrderSide, current_price: float):
        """Executes a trade based on the generated signal."""
        if signal == OrderSide.BUY:
            logger.info(f"Executing BUY signal at price ~{current_price}")
            try:
                order_result = await self._place_order(
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET, # Or LIMIT based on strategy
                    amount=self.trade_amount_base
                    # price=current_price # Add price for LIMIT orders
                )
                # Update current position state ONLY after successful order placement/confirmation
                # In simulation, this happens immediately. In live, might need to check order status.
                self.current_position = {
                    'side': OrderSide.BUY,
                    'entry_price': order_result.get('price', current_price), # Use actual fill price
                    'amount': order_result.get('filled', self.trade_amount_base), # Use actual filled amount
                    'order_id': order_result.get('id')
                }
                logger.info(f"Entered BUY position: Amount={self.current_position['amount']}, Entry Price={self.current_position['entry_price']}")
            except Exception as e:
                logger.error(f"Failed to execute BUY signal: {e}")
                # Reset state or handle error appropriately
                self.current_position = None

        elif signal == OrderSide.SELL: # Handle sell/short signal execution
            logger.info(f"Executing SELL signal at price ~{current_price}")
            # Implement short selling logic if applicable
            # This usually involves placing a sell order
            # Make sure the exchange and account support shorting
            try:
                # Example for shorting (adjust amount based on quote currency if needed)
                # amount_to_short = self.trade_amount_quote / current_price # Example if config is quote amount
                amount_to_short = self.trade_amount_base # Assuming config is base amount
                order_result = await self._place_order(
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET, # Or LIMIT
                    amount=amount_to_short
                )
                self.current_position = {
                    'side': OrderSide.SELL,
                    'entry_price': order_result.get('price', current_price),
                    'amount': order_result.get('filled', amount_to_short),
                    'order_id': order_result.get('id')
                }
                logger.info(f"Entered SELL position: Amount={self.current_position['amount']}, Entry Price={self.current_position['entry_price']}")
            except Exception as e:
                logger.error(f"Failed to execute SELL signal: {e}")
                self.current_position = None


    async def _run_iteration(self):
        """Runs a single iteration of the trading loop."""
        self.iteration += 1
        logger.info(f"{Fore.BLUE}----- Iteration {self.iteration} -----{Style.RESET_ALL}")

        # 1. Fetch Data
        # Fetch ticker first for current price
        ticker = await self._fetch_ticker()
        current_price = ticker.get('last')
        if current_price is None:
            logger.error("Could not fetch current price from ticker. Skipping iteration.")
            return # Cannot proceed without price

        logger.info(f"Current Price ({self.symbol}): {current_price}")

        # Fetch/update historical data
        await self._update_historical_data()
        # Fetch order book if needed by strategy
        # order_book = await self._fetch_order_book()

        # 2. Manage Open Positions (Check TP/SL first)
        await self._manage_open_position(current_price)

        # 3. Calculate Indicators
        indicators = self._calculate_indicators()

        # 4. Generate Trade Signal (only if no position is open)
        signal = self._generate_trade_signal(indicators, current_price)

        # 5. Execute Trade Signal
        if signal:
            await self._execute_trade_signal(signal, current_price)
        else:
            logger.debug("No new trade signal generated or position already open.")


    async def run(self):
        """Main asynchronous trading loop."""
        logger.info(f"Starting trading bot for {self.symbol}...")
        if self.simulation_mode:
            logger.warning(f"{Fore.YELLOW}--- RUNNING IN SIMULATION MODE ---{Style.RESET_ALL}")

        # Initial data load
        logger.info("Performing initial historical data load...")
        await self._update_historical_data() # Load more data initially
        logger.info(f"Initial data load complete. Buffer size: {len(self.historical_data)}")


        while True:
            start_time = time.monotonic()
            try:
                await self._run_iteration()

            except ccxt.AuthenticationError as e:
                logger.critical(f"Authentication error: {e}. Please check API keys. Exiting.")
                break # Exit loop on auth errors
            except ccxt.InsufficientFunds as e:
                 logger.error(f"Insufficient funds: {e}. Pausing trading for 60 seconds.")
                 await asyncio.sleep(60) # Pause
                 continue # Continue loop after pause
            except RETRYABLE_EXCEPTIONS as e:
                # These should have been handled by the decorator, but catch here as a fallback
                logger.error(f"API Error (should have been retried): {e}. Pausing briefly.")
                await asyncio.sleep(self.trade_loop_delay * 2) # Longer pause
                continue
            except ccxt.ExchangeError as e:
                # Handle specific non-retryable exchange errors if needed
                logger.error(f"Unhandled Exchange error: {e}. Analyzing...")
                # Example: Check for maintenance or specific error messages
                if "maintenance" in str(e).lower():
                    logger.warning("Exchange potentially in maintenance. Pausing for 5 minutes.")
                    await asyncio.sleep(300)
                    continue
                else:
                    logger.exception(f"Unhandled Exchange Error, stopping bot: {e}") # Log full traceback
                    break # Stop on potentially serious unhandled exchange errors
            except Exception as e:
                logger.exception(f"Unexpected error in main loop: {e}")
                # Consider if bot should stop or continue on general errors
                # break # Option: Stop on any unexpected error
                await asyncio.sleep(self.trade_loop_delay) # Option: Pause and continue

            # Calculate time taken and sleep accordingly
            end_time = time.monotonic()
            iteration_duration = end_time - start_time
            sleep_time = max(0, self.trade_loop_delay - iteration_duration)
            logger.debug(f"Iteration took {iteration_duration:.2f}s. Sleeping for {sleep_time:.2f}s.")
            await asyncio.sleep(sleep_time)


    async def close(self):
        """Gracefully closes the exchange connection."""
        if hasattr(self, 'exchange') and self.exchange:
            logger.info(f"Closing connection to {self.exchange.id}...")
            try:
                await self.exchange.close()
                logger.info("Exchange connection closed.")
            except Exception as e:
                logger.error(f"Error closing exchange connection: {e}")


async def main():
    """Main entry point for the asynchronous bot."""
    bot = None
    try:
        bot = ScalpingBot()
        await bot.run()
    except (FileNotFoundError, ValueError, TypeError, yaml.YAMLError) as e:
        # Config loading/validation errors already logged by ScalpingBot.__init__
        logger.critical(f"Critical initialization error: {e}. Exiting.")
        sys.exit(1)
    except ccxt.AuthenticationError as e:
        # Handled in run loop, but catch here if it happens during init
        logger.critical(f"Authentication error during initialization: {e}. Exiting.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping bot...")
    except Exception as e:
        logger.exception(f"Unhandled critical error: {e}. Exiting.")
        sys.exit(1)
    finally:
        if bot:
            await bot.close()
        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    # Ensure the script is run with Python 3.7+ for asyncio features
    if sys.version_info < (3, 7):
        sys.stderr.write("This script requires Python 3.7 or later.
")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it happens before the main loop's handler is active
        logger.info("Keyboard interrupt detected during startup/shutdown.")
