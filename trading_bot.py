"""
Trading Bot Core Module

This module implements the main trading bot logic, including:
- Exchange connection and market data retrieval
- Technical analysis and signal generation
- Order execution and management
- Position tracking and risk management
"""

import json
import logging
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Union, Any

import ccxt
import numpy as np
import pandas as pd

from indicators import calculate_indicators, calculate_signal
from strategies import evaluate_strategies, calculate_momentum_divergence_strategy, calculate_multi_timeframe_trend_strategy, calculate_support_resistance_breakout_strategy
from risk_management import calculate_position_size, calculate_dynamic_stop_loss, calculate_take_profit, update_trailing_stop, check_max_drawdown, adjust_risk_after_losses
from utils import parse_timeframe, setup_ccxt_exchange, retry_api_call

# Configure logger
logger = logging.getLogger("trading_bot")


class TradingBot:
    """
    Main trading bot class that handles the entire trading logic flow:
    - Loading configuration
    - Connecting to exchange
    - Retrieving market data
    - Analyzing with technical indicators
    - Executing trades based on signals
    - Managing risk and positions
    """

    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the trading bot with the given configuration.
        
        Args:
            config_file: Path to the configuration file
        """
        self.logger = logger
        self.config_file = config_file
        self.config = self.load_config()
        self.state_file = "bot_state.json"
        self.state = self.load_state()
        
        # Initialize exchange connection
        self.exchange_id = self.config.get("exchange", "bybit")
        self.exchange = self.setup_exchange()
        
        # Market and position information
        self.symbol = self.config.get("symbol", "BTC/USDT:USDT")
        self.timeframe = self.config.get("timeframe", "15m")
        self.market_info = None
        self.current_position = None
        self.candles_df = None
        self.precision = {
            "price": self.config.get("advanced", {}).get("price_precision", 2),
            "amount": self.config.get("advanced", {}).get("amount_precision", 6)
        }
        
        # Initialize exchange and fetch market data
        self.initialize()

    def load_config(self) -> Dict:
        """
        Load configuration from JSON file.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file {self.config_file} not found")
            # Create default config if file doesn't exist
            default_config = {
                "exchange": "bybit",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "15m",
                "api_key": "",
                "api_secret": "",
                "test_mode": True
            }
            with open(self.config_file, "w") as f:
                json.dump(default_config, f, indent=4)
            self.logger.info(f"Created default configuration file {self.config_file}")
            return default_config
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file {self.config_file}")
            raise

    def load_state(self) -> Dict:
        """
        Load bot state from JSON file.
        
        Returns:
            Dict: Bot state dictionary
        """
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.logger.info(f"State loaded from {self.state_file}")
            return state
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize with default state if file doesn't exist or is invalid
            default_state = {
                "positions": {},
                "orders": {},
                "trades": [],
                "last_update": 0
            }
            self.save_state(default_state)
            return default_state

    def save_state(self, state: Optional[Dict] = None) -> None:
        """
        Save bot state to JSON file.
        
        Args:
            state: State dictionary to save (uses self.state if None)
        """
        if state is None:
            state = self.state
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
            self.logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def setup_exchange(self) -> ccxt.Exchange:
        """
        Set up the CCXT exchange instance.
        
        Returns:
            ccxt.Exchange: Configured exchange instance
        """
        # Check for environment variables specific to the exchange first
        env_prefix = self.exchange_id.upper()
        api_key_env = os.getenv(f"{env_prefix}_API_KEY")
        api_secret_env = os.getenv(f"{env_prefix}_API_SECRET")
        
        # Use environment variables if available, otherwise use config values
        api_key = api_key_env or self.config.get("api_key") or os.getenv("API_KEY")
        api_secret = api_secret_env or self.config.get("api_secret") or os.getenv("API_SECRET")
        
        # Default options for the exchange
        options = {}
        params = {}
        
        # Add exchange-specific options
        if self.exchange_id == "bybit":
            options["defaultType"] = "swap"
            if self.config.get("test_mode", True):
                params["testnet"] = True
                
        self.logger.info(f"Setting up {self.exchange_id} exchange connection")
        if not api_key or not api_secret:
            self.logger.warning("API credentials not found. Running in read-only mode.")
        else:
            self.logger.info("API credentials found. Full trading functionality enabled.")
        
        # Create exchange instance with retry mechanism
        exchange = setup_ccxt_exchange(
            self.exchange_id,
            api_key,
            api_secret,
            options=options,
            params=params
        )
        
        if exchange:
            self.logger.info(f"Connected to {exchange.id} exchange")
            # Load markets to get trading info
            retry_api_call(exchange.load_markets)
            return exchange
        else:
            self.logger.error(f"Failed to connect to {self.exchange_id} exchange")
            raise ConnectionError(f"Failed to connect to {self.exchange_id} exchange")

    def initialize(self) -> None:
        """Initialize exchange connection and load market data"""
        try:
            # Fetch market information for the symbol
            self.market_info = self.exchange.market(self.symbol)
            self.logger.info(f"Market info loaded for {self.symbol}")
            
            # Update precision settings from market info if available
            if "precision" in self.market_info:
                if "price" in self.market_info["precision"]:
                    self.precision["price"] = self.market_info["precision"]["price"]
                if "amount" in self.market_info["precision"]:
                    self.precision["amount"] = self.market_info["precision"]["amount"]
            
            # Fetch current position
            self.update_position()
            
            # Fetch initial candles
            self.update_candles()
            
            self.logger.info(f"Bot initialized successfully for {self.symbol}")
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    def update_candles(self) -> None:
        """Fetch and update OHLCV candles data"""
        try:
            # Determine how many candles we need
            candles_required = self.config.get("advanced", {}).get("candles_required", 100)
            
            # Fetch candles with retry mechanism
            ohlcv = retry_api_call(
                self.exchange.fetch_ohlcv,
                self.symbol,
                timeframe=self.timeframe,
                limit=candles_required
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            self.candles_df = df
            self.logger.debug(f"Updated {len(df)} candles for {self.symbol}")
            
            # Calculate indicators
            self.candles_df = calculate_indicators(
                self.candles_df, 
                self.config.get("strategy", {}).get("indicators", {})
            )
        except Exception as e:
            self.logger.error(f"Error updating candles: {e}")
            raise

    def update_position(self) -> None:
        """Update current position information from exchange"""
        try:
            positions = retry_api_call(
                self.exchange.fetch_positions,
                [self.symbol]
            ) or []
            
            # Find the position for our symbol
            for position in positions:
                if position["symbol"] == self.symbol:
                    if float(position["contracts"]) > 0:
                        self.current_position = {
                            "side": "long",
                            "amount": float(position["contracts"]),
                            "entry_price": float(position["entryPrice"]),
                            "pnl": float(position["unrealizedPnl"]),
                            "liquidation_price": float(position.get("liquidationPrice", 0))
                        }
                        break
                    elif float(position["contracts"]) < 0:
                        self.current_position = {
                            "side": "short",
                            "amount": abs(float(position["contracts"])),
                            "entry_price": float(position["entryPrice"]),
                            "pnl": float(position["unrealizedPnl"]),
                            "liquidation_price": float(position.get("liquidationPrice", 0))
                        }
                        break
            else:
                self.current_position = None
            
            # Update state
            if self.current_position:
                self.state["positions"][self.symbol] = self.current_position
            elif self.symbol in self.state["positions"]:
                del self.state["positions"][self.symbol]
            
            self.save_state()
            self.logger.debug(f"Position updated: {self.current_position}")
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")

    def calculate_position_size(self, price: float, risk_pct: float = None) -> float:
        """
        Calculate position size based on risk management settings.
        
        Args:
            price: Current price for the asset
            risk_pct: Risk percentage override (uses config value if None)
            
        Returns:
            float: Position size in base currency
        """
        # Get risk management settings
        risk_settings = self.config.get("risk_management", {})
        if risk_pct is None:
            risk_pct = risk_settings.get("max_risk_per_trade_pct", 1.0)
        
        try:
            # Fetch account balance
            balance = retry_api_call(
                self.exchange.fetch_balance
            )
            
            quote_currency = self.symbol.split('/')[1].split(':')[0]
            available_balance = float(balance.get(quote_currency, {}).get("free", 0))
            
            # Calculate position size based on percentage of balance
            position_size_pct = risk_settings.get("position_size_pct", 1.0)
            position_value = available_balance * (position_size_pct / 100)
            
            # If ATR position sizing is enabled, adjust based on ATR
            if risk_settings.get("use_atr_position_sizing", False) and "atr" in self.candles_df.columns:
                atr = self.candles_df["atr"].iloc[-1]
                stop_loss_pct = None
                
                # Get stop loss configuration
                sl_config = risk_settings.get("stop_loss", {})
                if sl_config.get("enabled", True):
                    sl_mode = sl_config.get("mode", "atr")
                    if sl_mode == "atr":
                        atr_multiplier = sl_config.get("atr_multiplier", 2.0)
                        stop_loss_pct = (atr * atr_multiplier / price) * 100
                    else:
                        stop_loss_pct = sl_config.get("fixed_pct", 2.0)
                else:
                    # Default stop loss percentage if not configured
                    stop_loss_pct = 2.0
                
                # Calculate risk-adjusted position size
                risk_amount = available_balance * (risk_pct / 100)
                if stop_loss_pct > 0:
                    position_value = min(position_value, risk_amount / (stop_loss_pct / 100))
            
            # Calculate actual amount in base currency
            amount = position_value / price
            
            # Round down to meet exchange precision requirements
            decimal_places = self.precision["amount"]
            amount = Decimal(str(amount)).quantize(
                Decimal('0.' + '0' * decimal_places),
                rounding=ROUND_DOWN
            )
            
            # Check against minimum amount
            min_amount = self.config.get("advanced", {}).get("min_amount", 0.001)
            if float(amount) < min_amount:
                self.logger.warning(
                    f"Calculated position size {float(amount)} is below minimum {min_amount}. "
                    f"Using minimum amount."
                )
                amount = Decimal(str(min_amount))
                
            return float(amount)
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Return a safe minimal value
            return self.config.get("advanced", {}).get("min_amount", 0.001)

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price based on configuration.
        
        Args:
            entry_price: Entry price for the position
            side: Position side ('long' or 'short')
            
        Returns:
            float: Stop loss price
        """
        risk_settings = self.config.get("risk_management", {})
        sl_config = risk_settings.get("stop_loss", {})
        
        if not sl_config.get("enabled", True):
            return 0.0
        
        sl_mode = sl_config.get("mode", "atr")
        
        if sl_mode == "atr" and "atr" in self.candles_df.columns:
            atr = self.candles_df["atr"].iloc[-1]
            atr_multiplier = sl_config.get("atr_multiplier", 2.0)
            
            if side == "long":
                sl_price = entry_price - (atr * atr_multiplier)
            else:
                sl_price = entry_price + (atr * atr_multiplier)
        else:
            # Fixed percentage
            fixed_pct = sl_config.get("fixed_pct", 2.0)
            
            if side == "long":
                sl_price = entry_price * (1 - fixed_pct / 100)
            else:
                sl_price = entry_price * (1 + fixed_pct / 100)
        
        # Round to price precision
        decimal_places = self.precision["price"]
        sl_price = Decimal(str(sl_price)).quantize(
            Decimal('0.' + '0' * decimal_places),
            rounding=ROUND_DOWN if side == "long" else ROUND_DOWN  # Ensure conservative rounding
        )
        
        return float(sl_price)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price based on configuration.
        
        Args:
            entry_price: Entry price for the position
            side: Position side ('long' or 'short')
            
        Returns:
            float: Take profit price
        """
        risk_settings = self.config.get("risk_management", {})
        tp_config = risk_settings.get("take_profit", {})
        
        if not tp_config.get("enabled", True):
            return 0.0
        
        tp_mode = tp_config.get("mode", "atr")
        
        if tp_mode == "atr" and "atr" in self.candles_df.columns:
            atr = self.candles_df["atr"].iloc[-1]
            atr_multiplier = tp_config.get("atr_multiplier", 4.0)
            
            if side == "long":
                tp_price = entry_price + (atr * atr_multiplier)
            else:
                tp_price = entry_price - (atr * atr_multiplier)
        else:
            # Fixed percentage
            fixed_pct = tp_config.get("fixed_pct", 4.0)
            
            if side == "long":
                tp_price = entry_price * (1 + fixed_pct / 100)
            else:
                tp_price = entry_price * (1 - fixed_pct / 100)
        
        # Round to price precision
        decimal_places = self.precision["price"]
        tp_price = Decimal(str(tp_price)).quantize(
            Decimal('0.' + '0' * decimal_places),
            rounding=ROUND_DOWN if side == "short" else ROUND_DOWN  # Ensure conservative rounding
        )
        
        return float(tp_price)

    def execute_entry(self, side: str, trade_params: Dict = None) -> Dict:
        """
        Execute an entry order for the given side.
        
        Args:
            side: Order side ('buy' for long, 'sell' for short)
            trade_params: Additional trade parameters including order type, price, etc.
            
        Returns:
            Dict: Order result information
        """
        try:
            # Get current market price
            ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
            current_price = ticker["last"]
            
            # Calculate position size
            amount = self.calculate_position_size(current_price)
            
            # Calculate stop loss and take profit
            entry_side = "long" if side == "buy" else "short"
            sl_price = self.calculate_stop_loss(current_price, entry_side)
            tp_price = self.calculate_take_profit(current_price, entry_side)
            
            # Execute order
            order = retry_api_call(
                self.exchange.create_market_order,
                self.symbol,
                side,
                amount
            )
            
            # Wait for order to be filled
            time.sleep(1)
            
            # Update order with actual fill price and amount
            filled_order = retry_api_call(
                self.exchange.fetch_order,
                order["id"],
                self.symbol
            )
            
            # Update position
            self.update_position()
            
            # If stop loss or take profit is enabled, place those orders
            if sl_price > 0:
                sl_side = "sell" if entry_side == "long" else "buy"
                sl_order = retry_api_call(
                    self.exchange.create_order,
                    self.symbol,
                    "stop",
                    sl_side,
                    amount,
                    price=sl_price,
                    params={"stopPrice": sl_price}
                )
                self.logger.info(
                    f"Placed stop loss order at {sl_price} for {amount} {self.symbol}"
                )
                
                # Save stop loss order to state
                self.state["orders"]["sl"] = {
                    "id": sl_order["id"],
                    "price": sl_price,
                    "amount": amount,
                    "side": sl_side
                }
            
            if tp_price > 0:
                tp_side = "sell" if entry_side == "long" else "buy"
                tp_order = retry_api_call(
                    self.exchange.create_order,
                    self.symbol,
                    "limit",
                    tp_side,
                    amount,
                    price=tp_price
                )
                self.logger.info(
                    f"Placed take profit order at {tp_price} for {amount} {self.symbol}"
                )
                
                # Save take profit order to state
                self.state["orders"]["tp"] = {
                    "id": tp_order["id"],
                    "price": tp_price,
                    "amount": amount,
                    "side": tp_side
                }
            
            # Save filled order details to state
            if filled_order["status"] == "closed":
                entry_price = filled_order["price"]
                filled_amount = filled_order["filled"]
                
                entry_trade = {
                    "symbol": self.symbol,
                    "entry_side": entry_side,
                    "entry_price": entry_price,
                    "amount": filled_amount,
                    "timestamp": int(time.time() * 1000),
                    "sl_price": sl_price,
                    "tp_price": tp_price
                }
                
                self.state["positions"][self.symbol] = {
                    "side": entry_side,
                    "amount": filled_amount,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price
                }
                
                self.state["current_trade"] = entry_trade
                self.save_state()
                
                self.logger.info(
                    f"Opened {entry_side} position of {filled_amount} {self.symbol} at {entry_price}"
                )
                
                return {
                    "success": True,
                    "side": entry_side,
                    "amount": filled_amount,
                    "price": entry_price
                }
            else:
                self.logger.warning(f"Order not fully filled: {filled_order}")
                return {
                    "success": False,
                    "message": "Order not fully filled"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing entry: {e}")
            return {
                "success": False,
                "message": str(e)
            }

    def execute_exit(self) -> Dict:
        """
        Execute an exit order for the current position.
        
        Returns:
            Dict: Order result information
        """
        if not self.current_position:
            self.logger.warning("No position to exit")
            return {
                "success": False,
                "message": "No position to exit"
            }
        
        try:
            # Cancel any existing SL/TP orders
            if "sl" in self.state.get("orders", {}):
                try:
                    sl_order_id = self.state["orders"]["sl"]["id"]
                    retry_api_call(
                        self.exchange.cancel_order,
                        sl_order_id,
                        self.symbol
                    )
                    self.logger.info(f"Canceled stop loss order {sl_order_id}")
                except Exception as e:
                    self.logger.warning(f"Error canceling stop loss order: {e}")
            
            if "tp" in self.state.get("orders", {}):
                try:
                    tp_order_id = self.state["orders"]["tp"]["id"]
                    retry_api_call(
                        self.exchange.cancel_order,
                        tp_order_id,
                        self.symbol
                    )
                    self.logger.info(f"Canceled take profit order {tp_order_id}")
                except Exception as e:
                    self.logger.warning(f"Error canceling take profit order: {e}")
            
            # Execute market order to close position
            side = "sell" if self.current_position["side"] == "long" else "buy"
            amount = self.current_position["amount"]
            
            order = retry_api_call(
                self.exchange.create_market_order,
                self.symbol,
                side,
                amount
            )
            
            # Wait for order to be filled
            time.sleep(1)
            
            # Update order with actual fill price
            filled_order = retry_api_call(
                self.exchange.fetch_order,
                order["id"],
                self.symbol
            )
            
            if filled_order["status"] == "closed":
                exit_price = filled_order["price"]
                filled_amount = filled_order["filled"]
                
                # Calculate PnL
                entry_price = self.current_position["entry_price"]
                if self.current_position["side"] == "long":
                    pnl = (exit_price - entry_price) * filled_amount
                    pnl_pct = (exit_price / entry_price - 1) * 100
                else:  # short
                    pnl = (entry_price - exit_price) * filled_amount
                    pnl_pct = (1 - exit_price / entry_price) * 100
                
                # Add trade to history
                if "current_trade" in self.state:
                    current_trade = self.state["current_trade"]
                    current_trade["exit_price"] = exit_price
                    current_trade["exit_timestamp"] = int(time.time() * 1000)
                    current_trade["pnl"] = pnl
                    current_trade["pnl_pct"] = pnl_pct
                    
                    self.state["trades"].append(current_trade)
                    del self.state["current_trade"]
                
                # Clear position from state
                if self.symbol in self.state["positions"]:
                    del self.state["positions"][self.symbol]
                
                # Clear orders from state
                self.state["orders"] = {}
                
                self.save_state()
                
                # Update position
                self.update_position()
                
                self.logger.info(
                    f"Closed {self.current_position['side']} position of {filled_amount} {self.symbol} "
                    f"at {exit_price} with PnL: {pnl:.2f} ({pnl_pct:.2f}%)"
                )
                
                return {
                    "success": True,
                    "amount": filled_amount,
                    "price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                }
            else:
                self.logger.warning(f"Exit order not fully filled: {filled_order}")
                return {
                    "success": False,
                    "message": "Exit order not fully filled"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            return {
                "success": False,
                "message": str(e)
            }

    def should_enter_trade(self) -> Tuple[bool, str, Dict]:
        """
        Determine if we should enter a trade based on advanced strategies.
        
        Returns:
            Tuple[bool, str, Dict]: (should_enter, side, trade_parameters)
        """
        # Don't enter if we already have a position
        if self.current_position:
            return False, "", {}
        
        # Try to load strategy configuration
        strategy_config_file = "strategy_config.json"
        try:
            with open(strategy_config_file, "r") as f:
                strategy_config = json.load(f)
                
            # Check if advanced strategies are enabled
            active_strategy = strategy_config.get("active_strategy", "auto")
            self.logger.info(f"Using strategy mode: {active_strategy}")
            
            # If using only basic indicators, fall back to original method
            if active_strategy == "basic":
                raise FileNotFoundError("Using basic indicator mode")
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Fall back to basic indicator signals if strategy config not available
            self.logger.warning(f"Using basic indicators: {str(e)}")
            
            # Get strategy settings
            strategy = self.config.get("strategy", {})
            indicators = strategy.get("indicators", {})
            entry_threshold = strategy.get("entry_threshold", 0.5)
            
            # Get volume filter settings
            volume_filter = strategy.get("volume_filter", {})
            if volume_filter.get("enabled", False):
                min_volume = volume_filter.get("min_24h_volume_usd", 1000000)
                ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
                volume_usd = ticker["quoteVolume"]
                
                if volume_usd < min_volume:
                    self.logger.debug(
                        f"Volume filter rejected trade: {volume_usd:.2f} < {min_volume:.2f}"
                    )
                    return False, "", {}
            
            # Calculate signal
            signal_strength, signal_direction = calculate_signal(
                self.candles_df,
                indicators,
                threshold=entry_threshold
            )
            
            # Check if signal is strong enough to enter
            if signal_strength >= entry_threshold and signal_direction in ["long", "short"]:
                self.logger.info(
                    f"Entry signal: {signal_direction} with strength {signal_strength:.2f}"
                )
                return True, signal_direction, {}
            
            return False, "", {}
        
        # Apply global filters first
        filters = strategy_config.get("filters", {})
        
        # Check volume filter
        if filters.get("volume_filter", {}).get("enabled", True):
            min_volume = filters["volume_filter"].get("min_24h_volume_usd", 5000000)
            ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
            volume_usd = ticker["quoteVolume"]
            
            if volume_usd < min_volume:
                self.logger.debug(
                    f"Volume filter rejected trade: {volume_usd:.2f} < {min_volume:.2f}"
                )
                return False, "", {}
        
        # Check volatility filter
        if filters.get("volatility_filter", {}).get("enabled", True):
            if "atr" not in self.candles_df.columns:
                self.logger.warning("ATR indicator missing for volatility filter")
            else:
                current_atr = self.candles_df["atr"].iloc[-1]
                current_price = self.candles_df["close"].iloc[-1]
                atr_pct = (current_atr / current_price) * 100
                
                min_atr_pct = filters["volatility_filter"].get("min_atr_pct", 0.3)
                max_atr_pct = filters["volatility_filter"].get("max_atr_pct", 5.0)
                
                if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                    self.logger.debug(
                        f"Volatility filter rejected trade: ATR {atr_pct:.2f}% outside range [{min_atr_pct}%, {max_atr_pct}%]"
                    )
                    return False, "", {}
        
        # Check time filter
        if filters.get("time_filter", {}).get("enabled", False):
            import datetime
            current_hour = datetime.datetime.now().hour
            blackout_hours = filters["time_filter"].get("blackout_hours", [])
            
            if current_hour in blackout_hours:
                self.logger.debug(
                    f"Time filter rejected trade: Current hour {current_hour} in blackout hours {blackout_hours}"
                )
                return False, "", {}
        
        # Get higher timeframe data if available
        higher_tf = strategy_config.get("multi_timeframe_trend", {}).get("higher_timeframe", "1h")
        higher_tf_df = None
        
        try:
            # Fetch higher timeframe candles
            higher_tf_ohlcv = retry_api_call(
                self.exchange.fetch_ohlcv,
                self.symbol,
                timeframe=higher_tf,
                limit=100
            )
            
            # Convert to DataFrame
            higher_tf_df = pd.DataFrame(
                higher_tf_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # Convert timestamp to datetime
            higher_tf_df["timestamp"] = pd.to_datetime(higher_tf_df["timestamp"], unit="ms")
            higher_tf_df.set_index("timestamp", inplace=True)
            
            # Calculate indicators for higher timeframe
            higher_tf_df = calculate_indicators(
                higher_tf_df, 
                self.config.get("strategy", {}).get("indicators", {})
            )
            self.logger.debug(f"Successfully loaded higher timeframe data ({higher_tf})")
        except Exception as e:
            self.logger.error(f"Error fetching higher timeframe data: {e}")
            higher_tf_df = self.candles_df.copy()  # Fallback to current timeframe
        
        # Evaluate strategies
        strategy_signal = evaluate_strategies(
            self.candles_df,
            higher_tf_df,
            strategy_config
        )
        
        # Get signal threshold
        threshold = strategy_config.get("strategy_threshold", 0.7)
        
        # Check if we have a valid signal
        if strategy_signal["signal"] >= threshold and strategy_signal["direction"]:
            self.logger.info(
                f"Entry signal detected ({strategy_signal['strategy']}): "
                f"{strategy_signal['direction']} with strength {strategy_signal['signal']:.2f}"
            )
            return True, strategy_signal["direction"], strategy_signal["parameters"]
        
        return False, "", {}

    def should_exit_trade(self) -> bool:
        """
        Determine if we should exit the current position.
        
        Returns:
            bool: True if should exit, False otherwise
        """
        if not self.current_position:
            return False
        
        # Get current position details
        position_side = self.current_position["side"]
        entry_price = self.current_position["entry_price"]
        
        # Get current price
        ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
        current_price = ticker["last"]
        
        # Try to load the strategy configuration
        strategy_config_file = "strategy_config.json"
        try:
            with open(strategy_config_file, "r") as f:
                strategy_config = json.load(f)
                
            # Check if advanced exit rules are enabled
            active_strategy = strategy_config.get("active_strategy", "auto")
            if active_strategy == "basic":
                raise FileNotFoundError("Using basic indicator mode for exit")
            
            # Get risk management settings
            risk_mgmt = strategy_config.get("risk_management", {})
            
            # Check if we have trade parameters in the position state
            if "parameters" in self.current_position:
                trade_params = self.current_position["parameters"]
                
                # Check take profit level
                if "take_profit" in trade_params:
                    tp_price = trade_params["take_profit"]
                    if (position_side == "long" and current_price >= tp_price) or \
                       (position_side == "short" and current_price <= tp_price):
                        self.logger.info(f"Exit signal: Take profit target reached at {tp_price}")
                        return True
                
                # Check stop loss level
                if "stop_loss" in trade_params:
                    sl_price = trade_params["stop_loss"]
                    if (position_side == "long" and current_price <= sl_price) or \
                       (position_side == "short" and current_price >= sl_price):
                        self.logger.info(f"Exit signal: Stop loss triggered at {sl_price}")
                        return True
                
                # Check false breakout exit (for support/resistance strategy)
                if trade_params.get("strategy") == "support_resistance_breakout":
                    if "false_breakout_level" in trade_params:
                        fb_level = trade_params["false_breakout_level"]
                        if (position_side == "long" and current_price < fb_level) or \
                           (position_side == "short" and current_price > fb_level):
                            self.logger.info(f"Exit signal: False breakout detected at {fb_level}")
                            return True
                
                # Check time-based exit
                if "entry_time" in trade_params and "max_time_in_trade" in trade_params:
                    entry_time = trade_params["entry_time"]
                    max_time = trade_params["max_time_in_trade"]
                    current_time = time.time() * 1000  # Current time in ms
                    
                    if (current_time - entry_time) / (60 * 1000) > max_time:  # Convert to minutes
                        self.logger.info(f"Exit signal: Maximum time in trade ({max_time} candles) reached")
                        return True
                
                # Check trailing stop
                if risk_mgmt.get("trailing_stop", {}).get("enabled", True):
                    if "trailing_stop_price" in self.current_position:
                        ts_price = self.current_position["trailing_stop_price"]
                        if (position_side == "long" and current_price <= ts_price) or \
                           (position_side == "short" and current_price >= ts_price):
                            self.logger.info(f"Exit signal: Trailing stop triggered at {ts_price}")
                            return True
            
            # Check multi-timeframe trend exit signals
            if trade_params.get("strategy") == "multi_timeframe_trend" and "exit_indicator" in trade_params:
                if trade_params["exit_indicator"] == "Higher TF EMA 50 cross":
                    # Get higher timeframe data
                    higher_tf = strategy_config.get("multi_timeframe_trend", {}).get("higher_timeframe", "1h")
                    
                    try:
                        # Fetch higher timeframe candles
                        higher_tf_ohlcv = retry_api_call(
                            self.exchange.fetch_ohlcv,
                            self.symbol,
                            timeframe=higher_tf,
                            limit=100
                        )
                        
                        # Convert to DataFrame
                        higher_tf_df = pd.DataFrame(
                            higher_tf_ohlcv,
                            columns=["timestamp", "open", "high", "low", "close", "volume"]
                        )
                        
                        # Calculate EMA 50
                        if "ema_50" not in higher_tf_df.columns:
                            higher_tf_df["ema_50"] = higher_tf_df["close"].ewm(span=50, adjust=False).mean()
                        
                        # Check EMA cross condition
                        current_price = higher_tf_df["close"].iloc[-1]
                        current_ema50 = higher_tf_df["ema_50"].iloc[-1]
                        
                        if (position_side == "long" and current_price < current_ema50) or \
                           (position_side == "short" and current_price > current_ema50):
                            self.logger.info(f"Exit signal: Higher timeframe trend change detected")
                            return True
                    except Exception as e:
                        self.logger.error(f"Error checking higher timeframe exit: {e}")
                        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Fall back to basic exit rules
            self.logger.warning(f"Using basic exit rules: {str(e)}")
            
        # Basic exit rules (as a fallback)
        # Get strategy settings
        strategy = self.config.get("strategy", {})
        indicators = strategy.get("indicators", {})
        exit_threshold = strategy.get("exit_threshold", -0.3)
        
        # Calculate signal
        signal_strength, signal_direction = calculate_signal(
            self.candles_df,
            indicators,
            threshold=abs(exit_threshold)
        )
        
        # Check for exit signal (opposite to current position)
        if position_side == "long" and signal_direction == "short":
            if signal_strength >= abs(exit_threshold):
                self.logger.info(
                    f"Exit signal for long: short with strength {signal_strength:.2f}"
                )
                return True
        elif position_side == "short" and signal_direction == "long":
            if signal_strength >= abs(exit_threshold):
                self.logger.info(
                    f"Exit signal for short: long with strength {signal_strength:.2f}"
                )
                return True
        
        # Check for EMA cross exit condition if enabled
        if "ema_cross" in indicators and indicators["ema_cross"].get("use_for_exit", False):
            if "ema_cross" in self.candles_df.columns:
                ema_cross = self.candles_df["ema_cross"].iloc[-1]
                prev_ema_cross = self.candles_df["ema_cross"].iloc[-2] if len(self.candles_df) > 2 else 0
                
                if position_side == "long" and ema_cross == -1 and prev_ema_cross != -1:
                    self.logger.info("Exit signal for long: EMA cross turned bearish")
                    return True
                elif position_side == "short" and ema_cross == 1 and prev_ema_cross != 1:
                    self.logger.info("Exit signal for short: EMA cross turned bullish")
                    return True
        
        return False

    def check_break_even(self) -> None:
        """Check and adjust stop loss to break even if configured"""
        if not self.current_position:
            return
        
        risk_settings = self.config.get("risk_management", {})
        break_even = risk_settings.get("break_even", {})
        
        if not break_even.get("enabled", False):
            return
        
        # Get break even activation percentage
        activation_pct = break_even.get("activation_pct", 1.0)
        
        # Get current price
        ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
        current_price = ticker["last"]
        
        # Calculate current profit percentage
        entry_price = self.current_position["entry_price"]
        side = self.current_position["side"]
        
        if side == "long":
            profit_pct = (current_price / entry_price - 1) * 100
        else:  # short
            profit_pct = (1 - current_price / entry_price) * 100
        
        # If profit exceeds activation percentage, move stop loss to break even
        if profit_pct >= activation_pct:
            # Check if we have a stop loss order in state
            if "sl" in self.state.get("orders", {}):
                try:
                    # Cancel existing stop loss order
                    sl_order_id = self.state["orders"]["sl"]["id"]
                    retry_api_call(
                        self.exchange.cancel_order,
                        sl_order_id,
                        self.symbol
                    )
                    
                    # Place new stop loss order at break even
                    sl_side = "sell" if side == "long" else "buy"
                    amount = self.current_position["amount"]
                    
                    # Add a small buffer to break even price to account for fees
                    buffer_pct = 0.1  # 0.1% buffer
                    if side == "long":
                        break_even_price = entry_price * (1 + buffer_pct / 100)
                    else:
                        break_even_price = entry_price * (1 - buffer_pct / 100)
                    
                    # Round to price precision
                    decimal_places = self.precision["price"]
                    break_even_price = Decimal(str(break_even_price)).quantize(
                        Decimal('0.' + '0' * decimal_places),
                        rounding=ROUND_DOWN if side == "long" else ROUND_DOWN
                    )
                    
                    sl_order = retry_api_call(
                        self.exchange.create_order,
                        self.symbol,
                        "stop",
                        sl_side,
                        amount,
                        price=float(break_even_price),
                        params={"stopPrice": float(break_even_price)}
                    )
                    
                    # Update stop loss order in state
                    self.state["orders"]["sl"] = {
                        "id": sl_order["id"],
                        "price": float(break_even_price),
                        "amount": amount,
                        "side": sl_side
                    }
                    
                    self.save_state()
                    self.logger.info(
                        f"Moved stop loss to break even: {float(break_even_price)}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error adjusting stop loss to break even: {e}")

    def run_once(self) -> None:
        """
        Execute a single iteration of the trading bot logic.
        """
        try:
            # Update market data
            self.update_candles()
            
            # Update position information
            self.update_position()
            
            # Check if we should exit an existing position
            if self.current_position:
                if self.should_exit_trade():
                    self.execute_exit()
                else:
                    # If we're not exiting, check if we should move stop loss to break even
                    self.check_break_even()
                    
                    # Update trailing stop if active
                    self.update_trailing_stop()
            # Check if we should enter a new position
            else:
                should_enter, side, trade_params = self.should_enter_trade()
                if should_enter:
                    # Convert side to order side
                    order_side = "buy" if side == "long" else "sell"
                    
                    # Add current timestamp to trade parameters
                    if trade_params:
                        trade_params["entry_time"] = int(time.time() * 1000)
                        
                    # Execute entry with strategy parameters
                    self.execute_entry(order_side, trade_params)
            
            # Update state with timestamp
            self.state["last_update"] = int(time.time() * 1000)
            self.save_state()
            
        except Exception as e:
            self.logger.error(f"Error in run_once: {e}")
            
    def update_trailing_stop(self) -> None:
        """Update trailing stop if enabled and conditions are met"""
        if not self.current_position:
            return
            
        # Get current position details
        position_side = self.current_position["side"]
        entry_price = self.current_position["entry_price"]
        
        # Try to load strategy configuration
        try:
            with open("strategy_config.json", "r") as f:
                strategy_config = json.load(f)
                
            # Get trailing stop settings
            trailing_stop_config = strategy_config.get("risk_management", {}).get("trailing_stop", {})
            
            if not trailing_stop_config.get("enabled", True):
                return
                
            # Get current price
            ticker = retry_api_call(self.exchange.fetch_ticker, self.symbol)
            current_price = ticker["last"]
            
            # Get activation percentage and trail percentage
            activation_pct = trailing_stop_config.get("activation_pct", 1.5)
            trail_pct = trailing_stop_config.get("trail_pct", 0.75)
            
            # Get current trailing stop if it exists
            current_stop = self.current_position.get("trailing_stop_price", 0)
            
            # Default to initial stop loss if no trailing stop set yet
            if current_stop == 0:
                if "parameters" in self.current_position and "stop_loss" in self.current_position["parameters"]:
                    current_stop = self.current_position["parameters"]["stop_loss"]
                else:
                    # Calculate a default stop loss
                    if position_side == "long":
                        current_stop = entry_price * 0.95  # 5% below entry
                    else:
                        current_stop = entry_price * 1.05  # 5% above entry
            
            # Check if price has moved enough to activate trailing stop
            activation_threshold = 0.0
            if position_side == "long":
                activation_threshold = entry_price * (1 + activation_pct/100)
                
                if current_price >= activation_threshold:
                    # Calculate new trailing stop
                    new_stop = current_price * (1 - trail_pct/100)
                    
                    # Only update if new stop is higher than current
                    if new_stop > current_stop:
                        self.logger.info(f"Updated trailing stop: {current_stop} -> {new_stop}")
                        self.current_position["trailing_stop_price"] = new_stop
                        # Save position state
                        self.state["positions"][self.symbol] = self.current_position
                        self.save_state()
            else:  # short position
                activation_threshold = entry_price * (1 - activation_pct/100)
                
                if current_price <= activation_threshold:
                    # Calculate new trailing stop
                    new_stop = current_price * (1 + trail_pct/100)
                    
                    # Only update if new stop is lower than current
                    if new_stop < current_stop:
                        self.logger.info(f"Updated trailing stop: {current_stop} -> {new_stop}")
                        self.current_position["trailing_stop_price"] = new_stop
                        # Save position state
                        self.state["positions"][self.symbol] = self.current_position
                        self.save_state()
                        
        except (FileNotFoundError, json.JSONDecodeError):
            # Skip trailing stop update if config not available
            return

    def run(self) -> None:
        """
        Run the trading bot in continuous mode.
        """
        self.logger.info(f"Starting trading bot for {self.symbol} on {self.exchange_id}")
        
        loop_interval = self.config.get("loop_interval_seconds", 15)
        
        try:
            while True:
                start_time = time.time()
                
                self.run_once()
                
                # Calculate time to sleep
                elapsed = time.time() - start_time
                sleep_time = max(0, loop_interval - elapsed)
                
                if sleep_time > 0:
                    self.logger.debug(f"Sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Trading bot stopped by user")
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
            raise

    def run_backtest(self) -> Dict:
        """
        Run the trading bot in backtest mode using historical data.
        
        Returns:
            Dict: Backtest results
        """
        self.logger.info(f"Starting backtest for {self.symbol} on {self.exchange_id}")
        
        # Initialize backtest state
        backtest_state = {
            "trades": [],
            "current_position": None,
            "equity_curve": [],
            "starting_balance": 10000,  # Default starting balance
            "current_balance": 10000,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "peak_balance": 10000
        }
        
        try:
            # Fetch historical data for backtest period
            # (In a full implementation, this would fetch more historical data)
            historical_data = self.candles_df.copy()
            
            # Calculate indicators
            historical_data = calculate_indicators(
                historical_data,
                self.config.get("strategy", {}).get("indicators", {})
            )
            
            # Simulate trades on historical data
            for i in range(len(historical_data) - 1):
                # Skip the first few bars until we have enough data for indicators
                if i < 20:
                    continue
                
                # Get current candle data
                current_data = historical_data.iloc[:i+1]
                current_candle = current_data.iloc[-1]
                
                # Get strategy settings
                strategy = self.config.get("strategy", {})
                indicators = strategy.get("indicators", {})
                entry_threshold = strategy.get("entry_threshold", 0.5)
                exit_threshold = strategy.get("exit_threshold", -0.3)
                
                # Check for exit if in position
                if backtest_state["current_position"]:
                    signal_strength, signal_direction = calculate_signal(
                        current_data,
                        indicators,
                        threshold=abs(exit_threshold)
                    )
                    
                    position = backtest_state["current_position"]
                    should_exit = False
                    
                    # Check for exit signal (opposite to current position)
                    if position["side"] == "long" and signal_direction == "short":
                        if signal_strength >= abs(exit_threshold):
                            should_exit = True
                    elif position["side"] == "short" and signal_direction == "long":
                        if signal_strength >= abs(exit_threshold):
                            should_exit = True
                    
                    # Execute exit if needed
                    if should_exit:
                        exit_price = current_candle["close"]
                        entry_price = position["entry_price"]
                        amount = position["amount"]
                        
                        # Calculate PnL
                        if position["side"] == "long":
                            pnl = (exit_price - entry_price) * amount
                            pnl_pct = (exit_price / entry_price - 1) * 100
                        else:  # short
                            pnl = (entry_price - exit_price) * amount
                            pnl_pct = (1 - exit_price / entry_price) * 100
                        
                        # Update balance
                        backtest_state["current_balance"] += pnl
                        
                        # Record trade
                        trade = {
                            "entry_timestamp": position["timestamp"],
                            "exit_timestamp": current_candle.name.timestamp() * 1000,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "amount": amount,
                            "side": position["side"],
                            "pnl": pnl,
                            "pnl_pct": pnl_pct
                        }
                        backtest_state["trades"].append(trade)
                        
                        # Clear position
                        backtest_state["current_position"] = None
                        
                        self.logger.debug(
                            f"Backtest exit: {position['side']} at {exit_price} with PnL: {pnl:.2f} ({pnl_pct:.2f}%)"
                        )
                
                # Check for entry if not in position
                else:
                    signal_strength, signal_direction = calculate_signal(
                        current_data,
                        indicators,
                        threshold=entry_threshold
                    )
                    
                    # Check if signal is strong enough to enter
                    if signal_strength >= entry_threshold and signal_direction in ["long", "short"]:
                        entry_price = current_candle["close"]
                        
                        # Calculate position size
                        position_size_pct = self.config.get("risk_management", {}).get("position_size_pct", 1.0)
                        position_value = backtest_state["current_balance"] * (position_size_pct / 100)
                        amount = position_value / entry_price
                        
                        # Record position
                        backtest_state["current_position"] = {
                            "side": signal_direction,
                            "entry_price": entry_price,
                            "amount": amount,
                            "timestamp": current_candle.name.timestamp() * 1000
                        }
                        
                        self.logger.debug(
                            f"Backtest entry: {signal_direction} at {entry_price} with amount: {amount:.6f}"
                        )
                
                # Update equity curve
                backtest_state["equity_curve"].append({
                    "timestamp": current_candle.name.timestamp() * 1000,
                    "balance": backtest_state["current_balance"]
                })
                
                # Update drawdown metrics
                if backtest_state["current_balance"] > backtest_state["peak_balance"]:
                    backtest_state["peak_balance"] = backtest_state["current_balance"]
                
                current_drawdown = backtest_state["peak_balance"] - backtest_state["current_balance"]
                current_drawdown_pct = (current_drawdown / backtest_state["peak_balance"]) * 100
                
                if current_drawdown > backtest_state["max_drawdown"]:
                    backtest_state["max_drawdown"] = current_drawdown
                    backtest_state["max_drawdown_pct"] = current_drawdown_pct
            
            # Calculate backtest results
            total_trades = len(backtest_state["trades"])
            profitable_trades = sum(1 for trade in backtest_state["trades"] if trade["pnl"] > 0)
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate returns
            starting_balance = backtest_state["starting_balance"]
            final_balance = backtest_state["current_balance"]
            total_return = final_balance - starting_balance
            total_return_pct = (final_balance / starting_balance - 1) * 100
            
            results = {
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "win_rate": win_rate,
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "max_drawdown": backtest_state["max_drawdown"],
                "max_drawdown_pct": backtest_state["max_drawdown_pct"],
                "trades": backtest_state["trades"],
                "equity_curve": backtest_state["equity_curve"]
            }
            
            self.logger.info(f"Backtest completed with {total_trades} trades and {win_rate:.2f}% win rate")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return {"error": str(e)}