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
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from threading import Event
from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict

import ccxt
import numpy as np
import pandas as pd

from indicators import calculate_indicators, calculate_signal
from strategies import calculate_ehlers_supertrend_strategy
from risk_management import (
    calculate_position_size,
    calculate_dynamic_stop_loss,
    calculate_take_profit,
    update_trailing_stop
)
from utils import setup_ccxt_exchange, retry_api_call, safe_float

# Configure logger
logger = logging.getLogger("trading_bot")


class TradeParams(TypedDict, total=False):
    """Type definition for trade parameters"""
    symbol: str
    side: str
    size: float
    price: float
    stop_loss: float
    take_profit: float
    leverage: float
    reduce_only: bool
    order_type: str
    params: dict


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

    def __init__(
        self,
        config_file: str = "config.json",
        validate_only: bool = False,
        exchange: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        is_testnet: bool = False
    ):
        """
        Initialize the trading bot with the given configuration.
        
        Args:
            config_file: Path to the configuration file
            validate_only: If True, only validate the configuration without connecting to exchange
            exchange: Exchange name to override config
            api_key: API key to override config
            api_secret: API secret to override config
            is_testnet: Whether to use testnet (sandbox)
        """
        self.logger = logger
        self.config_file = config_file
        self.config = self.load_config()
        self.state_file = "bot_state.json"
        self.state = self.load_state()
        
        # Apply overrides
        if exchange:
            self.config["exchange"] = exchange
        if api_key:
            self.config["api_key"] = api_key
        if api_secret:
            self.config["api_secret"] = api_secret
        if is_testnet:
            self.config["test_mode"] = is_testnet
            
        # Initialize exchange and symbol information
        self.exchange_id = self.config.get("exchange", "bybit")
        self.symbol = self.config.get("symbol", "BTC/USDT:USDT")
        self.timeframe = self.config.get("timeframe", "15m")
        self.strategy_name = self.config.get("strategy", {}).get("active", "ehlers_supertrend")
        
        # Initialize state variables
        self.market_info = None
        self.current_positions = {}
        self.candles_df = None
        self.higher_timeframe_df = None
        self.last_update_time = 0
        self.last_analysis_time = 0
        self.trading_paused = False
        self.error_count = 0
        
        # Parse trading parameters
        self.precision = {
            "price": self.config.get("advanced", {}).get("price_precision", 2),
            "amount": self.config.get("advanced", {}).get("amount_precision", 6)
        }
        
        # Connect to exchange
        if not validate_only:
            self.exchange = self.setup_exchange()
            # Initialize market data and positions
            self.initialize()
        else:
            self.exchange = None
            self.logger.info("Running in validation mode - not connecting to exchange")

    def load_config(self) -> Dict:
        """
        Load configuration from JSON file
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading config from {self.config_file}: {e}")
            # Return default configuration
            self.logger.warning("Using default configuration")
            return {
                "exchange": "bybit",
                "symbol": "BTC/USDT:USDT",
                "timeframe": "15m",
                "api_key": "use_env_variable",
                "api_secret": "use_env_variable",
                "test_mode": True,
                "strategy": {
                    "active": "ehlers_supertrend",
                    "indicators": {
                        "rsi": {"window": 14, "overbought": 70, "oversold": 30},
                        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                        "bollinger_bands": {"window": 20, "std_dev": 2.0},
                        "atr": {"window": 14}
                    }
                },
                "risk_management": {
                    "max_open_positions": 1,
                    "position_size_pct": 1.0,
                    "stop_loss_pct": 2.0,
                    "take_profit_pct": 4.0,
                    "trailing_stop": {
                        "enabled": True,
                        "activation_pct": 1.0,
                        "trail_pct": 0.5
                    }
                },
                "advanced": {
                    "price_precision": 2,
                    "amount_precision": 6,
                    "candles_limit": 200,
                }
            }

    def load_state(self) -> Dict:
        """
        Load bot state from JSON file
        
        Returns:
            Dict: State dictionary
        """
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                return state
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error loading state from {self.state_file}: {e}")
            # Return default state structure
            return {
                "active": False,
                "last_update": 0,
                "symbols": [],
                "timeframes": [],
                "active_strategy": "",
                "balance": {
                    "last_checked": 0,
                    "total": 0,
                    "free": 0,
                    "used": 0,
                    "history": []
                },
                "positions": {},
                "orders": {"active": {}, "history": []},
                "trades": {"total": 0, "wins": 0, "losses": 0, "recent": []},
                "performance": {
                    "pnl_total": 0.0,
                    "pnl_percentage": 0.0,
                    "drawdown_current": 0.0,
                    "drawdown_max": 0.0
                },
                "strategy_stats": {},
                "market_conditions": {},
                "errors": {
                    "last_error": "",
                    "last_error_time": 0,
                    "error_count": 0,
                    "recent_errors": []
                },
                "config": {
                    "dry_run": True,
                    "exchange": self.exchange_id
                }
            }

    def save_state(self) -> bool:
        """
        Save bot state to JSON file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update timestamp
            self.state["last_update"] = int(time.time() * 1000)
            
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving state to {self.state_file}: {e}")
            return False

    def setup_exchange(self) -> ccxt.Exchange:
        """
        Set up and configure the CCXT exchange connection
        
        Returns:
            ccxt.Exchange: Configured exchange instance
        """
        api_key = self.config.get("api_key")
        api_secret = self.config.get("api_secret")
        
        # Check for environment variables if specified
        if api_key == "use_env_variable":
            api_key = os.environ.get(f"{self.exchange_id.upper()}_API_KEY")
        if api_secret == "use_env_variable":
            api_secret = os.environ.get(f"{self.exchange_id.upper()}_API_SECRET")
        
        # Check for required credentials
        if not api_key or not api_secret:
            self.logger.warning(f"API credentials not provided for {self.exchange_id}")
            if not self.config.get("dry_run", True):
                self.logger.error("Live trading requires API credentials")
                raise ValueError("API credentials required for live trading")
        
        # Setup exchange with error handling
        try:
            test_mode = self.config.get("test_mode", False)
            exchange = setup_ccxt_exchange(
                exchange_id=self.exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                testnet=test_mode
            )
            
            # Load markets for symbol info
            self.logger.info(f"Loading markets for {self.exchange_id}")
            exchange.load_markets()
            
            return exchange
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange {self.exchange_id}: {e}")
            raise

    def initialize(self) -> None:
        """Initialize the bot by fetching market data and positions"""
        try:
            # Fetch market information
            self.market_info = self.fetch_market_info()
            
            # Fetch current positions
            self.update_positions()
            
            # Fetch initial candles
            self.update_candles()
            
            # Update account balance
            self.update_balance()
            
            # Update bot state
            self.state["active"] = True
            self.state["active_strategy"] = self.strategy_name
            
            if self.symbol not in self.state["symbols"]:
                self.state["symbols"].append(self.symbol)
            
            if self.timeframe not in self.state["timeframes"]:
                self.state["timeframes"].append(self.timeframe)
            
            self.save_state()
            
            self.logger.info(f"Bot initialized for {self.symbol} on {self.exchange_id}")
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            raise

    def fetch_market_info(self) -> Dict:
        """
        Fetch market information for the configured symbol
        
        Returns:
            Dict: Market information
        """
        try:
            if self.symbol in self.exchange.markets:
                market = self.exchange.markets[self.symbol]
                
                # Add some derived convenience fields
                market["is_contract"] = market["swap"] or market["future"]
                market["is_linear"] = market.get("linear", False)
                market["is_inverse"] = market.get("inverse", False)
                
                # Create a descriptive string for this contract type
                if market["spot"]:
                    market["contract_type_str"] = "spot"
                elif market["swap"] and market["linear"]:
                    market["contract_type_str"] = "linear_perpetual"
                elif market["swap"] and market["inverse"]:
                    market["contract_type_str"] = "inverse_perpetual"
                elif market["future"] and market["linear"]:
                    market["contract_type_str"] = "linear_future"
                elif market["future"] and market["inverse"]:
                    market["contract_type_str"] = "inverse_future"
                else:
                    market["contract_type_str"] = "unknown"
                
                self.logger.info(f"Market info for {self.symbol}: {market['contract_type_str']}")
                
                # Safe extraction of limit values as Decimal
                try:
                    limits = market.get("limits", {})
                    market["min_amount_decimal"] = Decimal(str(limits.get("amount", {}).get("min", 0)))
                    market["max_amount_decimal"] = Decimal(str(limits.get("amount", {}).get("max", float('inf'))))
                    market["min_cost_decimal"] = Decimal(str(limits.get("cost", {}).get("min", 0)))
                    market["max_cost_decimal"] = Decimal(str(limits.get("cost", {}).get("max", float('inf'))))
                    
                    # Extract precision step info
                    precision = market.get("precision", {})
                    market["amount_precision_step_decimal"] = Decimal(str(10 ** -precision.get("amount", 8)))
                    market["price_precision_step_decimal"] = Decimal(str(10 ** -precision.get("price", 8)))
                    
                    # For contracts, extract contract size
                    if market["is_contract"]:
                        market["contract_size_decimal"] = Decimal(str(market.get("contractSize", 1)))
                    else:
                        market["contract_size_decimal"] = Decimal("1")
                except Exception as e:
                    self.logger.warning(f"Error converting market limits to Decimal: {e}")
                
                return market
            else:
                raise ValueError(f"Symbol {self.symbol} not found in {self.exchange_id} markets")
        except Exception as e:
            self.logger.error(f"Error fetching market info for {self.symbol}: {e}")
            raise

    def update_candles(self) -> pd.DataFrame:
        """
        Fetch and update OHLCV candles for the configured symbol and timeframe
        
        Returns:
            pd.DataFrame: DataFrame with candle data and indicators
        """
        try:
            # Determine how many candles to fetch
            candles_limit = self.config.get("advanced", {}).get("candles_limit", 200)
            
            # Fetch candles
            self.logger.info(f"Fetching {candles_limit} candles for {self.symbol} ({self.timeframe})")
            candles = retry_api_call(
                lambda: self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    limit=candles_limit
                )
            )
            
            if not candles or len(candles) == 0:
                self.logger.warning(f"No candles returned for {self.symbol} ({self.timeframe})")
                return self.candles_df
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Calculate indicators
            df = calculate_indicators(df, self.config)
            
            # Update class attribute
            self.candles_df = df
            self.last_update_time = time.time()
            
            return df
        except Exception as e:
            self.logger.error(f"Error updating candles: {e}")
            self.error_count += 1
            return self.candles_df  # Return previous data if available

    def update_higher_timeframe_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch and update OHLCV candles for a higher timeframe for multi-timeframe analysis
        
        Returns:
            pd.DataFrame: DataFrame with higher timeframe candle data
        """
        try:
            # Determine higher timeframe (e.g. if 15m, use 1h)
            timeframe_dict = {
                "1m": "15m",
                "5m": "1h",
                "15m": "4h",
                "30m": "6h",
                "1h": "1d",
                "4h": "1d",
                "1d": "1w"
            }
            higher_tf = timeframe_dict.get(self.timeframe, "1d")
            
            # Fetch candles
            self.logger.info(f"Fetching higher timeframe ({higher_tf}) candles for {self.symbol}")
            candles = retry_api_call(
                lambda: self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=higher_tf,
                    limit=100
                )
            )
            
            if not candles or len(candles) == 0:
                self.logger.warning(f"No higher timeframe candles returned for {self.symbol} ({higher_tf})")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Calculate indicators for higher timeframe
            df = calculate_indicators(df, self.config)
            
            # Update class attribute
            self.higher_timeframe_df = df
            
            return df
        except Exception as e:
            self.logger.error(f"Error updating higher timeframe data: {e}")
            return None

    def update_positions(self) -> Dict:
        """
        Fetch and update current positions from the exchange
        
        Returns:
            Dict: Updated positions dictionary
        """
        try:
            # Skip if in dry-run mode
            if self.config.get("dry_run", True):
                self.logger.info("Dry run mode - skipping position update")
                return self.current_positions
            
            # Fetch positions from exchange
            self.logger.info(f"Fetching positions for {self.symbol}")
            positions = retry_api_call(
                lambda: self.exchange.fetch_positions([self.symbol])
            )
            
            # Process positions
            position_dict = {}
            for pos in positions:
                symbol = pos["symbol"]
                contracts = safe_float(pos, "contracts", 0)
                if contracts > 0:
                    position_dict[symbol] = pos
                    self.logger.info(f"Found position: {symbol} - {pos['side']} {contracts} contracts at {pos['entryPrice']}")
            
            # Update state
            self.current_positions = position_dict
            self.state["positions"] = position_dict
            self.save_state()
            
            return position_dict
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            self.error_count += 1
            return self.current_positions

    def update_balance(self) -> Dict:
        """
        Fetch and update account balance
        
        Returns:
            Dict: Balance information
        """
        try:
            # Skip if in dry-run mode
            if self.config.get("dry_run", True):
                self.logger.info("Dry run mode - skipping balance update")
                return self.state["balance"]
            
            # Fetch balance from exchange
            self.logger.info("Fetching account balance")
            balance = retry_api_call(
                lambda: self.exchange.fetch_balance()
            )
            
            # Extract relevant information
            total = safe_float(balance, "total", 0)
            free = safe_float(balance, "free", 0)
            used = safe_float(balance, "used", 0)
            
            # Update state
            self.state["balance"]["last_checked"] = int(time.time() * 1000)
            self.state["balance"]["total"] = total
            self.state["balance"]["free"] = free
            self.state["balance"]["used"] = used
            
            # Update balance history (once per day)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            history = self.state["balance"].get("history", [])
            
            # Check if we already have an entry for today
            today_entry = next((item for item in history if item["date"] == today), None)
            if today_entry:
                today_entry["balance"] = total
            else:
                history.append({
                    "date": today,
                    "balance": total,
                    "timestamp": int(time.time() * 1000)
                })
                self.state["balance"]["history"] = history
            
            self.save_state()
            self.logger.info(f"Balance updated: {total} (Free: {free}, Used: {used})")
            
            return self.state["balance"]
        except Exception as e:
            self.logger.error(f"Error updating balance: {e}")
            self.error_count += 1
            return self.state["balance"]
            
    def get_balance(self) -> Dict:
        """
        Get the current account balance
        
        Returns:
            Dict: Balance information
        """
        return self.update_balance()
        
    def get_ticker(self) -> Dict:
        """
        Get ticker information for the current symbol
        
        Returns:
            Dict: Ticker information
        """
        try:
            if not self.exchange:
                # Return dummy data in case exchange is not available
                return {
                    "symbol": self.symbol,
                    "last": 0,
                    "bid": 0,
                    "ask": 0,
                    "high": 0,
                    "low": 0,
                    "volume": 0,
                    "timestamp": int(time.time() * 1000)
                }
                
            # Fetch ticker from exchange
            ticker = retry_api_call(
                lambda: self.exchange.fetch_ticker(self.symbol)
            )
            return ticker
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            return {}
            
    def start(self) -> bool:
        """
        Start the trading bot
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set active flag in state
            self.state["active"] = True
            self.save_state()
            
            self.logger.info("Trading bot started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            return False
            
    def stop(self) -> bool:
        """
        Stop the trading bot
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set active flag in state
            self.state["active"] = False
            self.save_state()
            
            self.logger.info("Trading bot stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {e}")
            return False

    def analyze_market(self) -> Dict:
        """
        Analyze market data using the selected strategy
        
        Returns:
            Dict: Analysis results
        """
        try:
            # Make sure we have the latest data
            if self.candles_df is None or time.time() - self.last_update_time > 60:
                self.update_candles()
            
            # For multi-timeframe strategies, also update higher timeframe data
            if self.strategy_name == "multi_timeframe_trend" or self.config.get("strategy", {}).get("use_higher_timeframe", False):
                self.update_higher_timeframe_data()
            
            # Calculate trading signal based on the selected strategy
            signal_strength, direction, params = self.calculate_signal()
            
            # Calculate risk parameters
            risk_params = self.calculate_risk_parameters(direction, params)
            
            # Update last analysis time
            self.last_analysis_time = time.time()
            
            # Create and return analysis result
            result = {
                "timestamp": int(time.time() * 1000),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy": self.strategy_name,
                "signal_strength": signal_strength,
                "direction": direction,
                "params": params,
                "risk_params": risk_params,
                "candles_count": len(self.candles_df) if self.candles_df is not None else 0,
                "current_price": self.get_current_price()
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing market: {e}")
            self.error_count += 1
            return {
                "timestamp": int(time.time() * 1000),
                "symbol": self.symbol,
                "strategy": self.strategy_name,
                "signal_strength": 0,
                "direction": "none",
                "error": str(e)
            }

    def calculate_signal(self) -> Tuple[float, str, Dict]:
        """
        Calculate trading signal using the selected strategy
        
        Returns:
            Tuple[float, str, Dict]: Signal strength, direction, and parameters
        """
        if self.candles_df is None:
            return 0, "none", {}
        
        strategy = self.strategy_name.lower()
        
        if strategy == "ehlers_supertrend":
            return calculate_ehlers_supertrend_strategy(
                self.candles_df,
                self.config.get("strategy", {}).get("ehlers_supertrend", {})
            )
        elif strategy == "supertrend":
            from indicators import calculate_supertrend_signal
            return calculate_supertrend_signal(
                self.candles_df, 
                self.config.get("strategy", {})
            )
        elif strategy == "macd_crossover":
            from indicators import calculate_macd_crossover_signal
            return calculate_macd_crossover_signal(
                self.candles_df,
                self.config.get("strategy", {})
            )
        elif strategy == "rsi_divergence":
            from indicators import calculate_rsi_divergence_signal
            return calculate_rsi_divergence_signal(
                self.candles_df,
                self.config.get("strategy", {})
            )
        elif strategy == "simple_crossover":
            from indicators import calculate_simple_crossover_signal
            return calculate_simple_crossover_signal(
                self.candles_df,
                self.config.get("strategy", {})
            )
        else:
            self.logger.warning(f"Unknown strategy: {strategy}, defaulting to supertrend")
            from indicators import calculate_supertrend_signal
            return calculate_supertrend_signal(
                self.candles_df,
                self.config.get("strategy", {})
            )

    def calculate_risk_parameters(self, direction: str, signal_params: Dict) -> Dict:
        """
        Calculate risk parameters for the trade
        
        Args:
            direction: Trade direction ('buy' or 'sell')
            signal_params: Parameters from the signal
            
        Returns:
            Dict: Risk parameters
        """
        if self.candles_df is None or direction == "none":
            return {}
        
        current_price = self.get_current_price()
        if current_price is None:
            return {}
        
        # Get ATR for position sizing and SL/TP
        atr = signal_params.get("atr")
        if atr is None:
            # Try to get ATR from DataFrame
            if "atr" in self.candles_df.columns:
                atr = self.candles_df["atr"].iloc[-1]
            else:
                # Use a simple volatility estimate if ATR not available
                high = self.candles_df["high"].iloc[-10:].max()
                low = self.candles_df["low"].iloc[-10:].min()
                close = self.candles_df["close"].iloc[-1]
                atr = (high - low) / 10
        
        # Set up risk parameters
        risk_config = self.config.get("risk_management", {})
        risk_percentage = risk_config.get("max_risk_per_trade_pct", 1.0) / 100.0
        
        # Get balance for position sizing
        if self.config.get("dry_run", True):
            # Use simulated balance in dry run
            balance = self.config.get("simulated_balance", 10000.0)
        else:
            # Use actual balance
            balance = self.state["balance"].get("total", 0)
        
        # Calculate SL and TP distances
        # Mode 1: Fixed percentage
        if risk_config.get("use_fixed_sl_tp", True):
            sl_percentage = risk_config.get("stop_loss_pct", 2.0) / 100.0
            tp_percentage = risk_config.get("take_profit_pct", 4.0) / 100.0
            
            if direction == "buy":
                sl_price = current_price * (1 - sl_percentage)
                tp_price = current_price * (1 + tp_percentage)
            else:
                sl_price = current_price * (1 + sl_percentage)
                tp_price = current_price * (1 - tp_percentage)
        
        # Mode 2: ATR-based
        else:
            sl_atr_mult = risk_config.get("sl_atr_mult", 1.5)
            tp_atr_mult = risk_config.get("tp_atr_mult", 3.0)
            
            if direction == "buy":
                sl_price = current_price - (atr * sl_atr_mult)
                tp_price = current_price + (atr * tp_atr_mult)
            else:
                sl_price = current_price + (atr * sl_atr_mult)
                tp_price = current_price - (atr * tp_atr_mult)
        
        # Calculate position size
        if risk_config.get("use_atr_position_sizing", False) and atr:
            # Risk-based position sizing using ATR
            risk_amount = balance * risk_percentage
            position_size = risk_amount / atr if atr > 0 else 0
        else:
            # Simple percentage of balance
            position_pct = risk_config.get("position_size_pct", 1.0) / 100.0
            position_size = (balance * position_pct) / current_price
        
        # Round position size to market precision
        if self.market_info:
            precision = self.market_info.get("precision", {}).get("amount", 8)
            min_amount = self.market_info.get("limits", {}).get("amount", {}).get("min", 0)
            
            # Round down to precision
            position_size = np.floor(position_size * 10**precision) / 10**precision
            
            # Check minimum size
            if position_size < min_amount:
                position_size = 0
                self.logger.warning(f"Calculated position size {position_size} is below minimum {min_amount}")
        
        # Calculate leverage (if applicable)
        leverage = 1.0
        if self.market_info and self.market_info.get("is_contract", False):
            leverage = risk_config.get("leverage", 1.0)
            # Adjust position size for leverage
            position_size = position_size * leverage
        
        # Trailing stop parameters
        trailing_stop = risk_config.get("trailing_stop", {})
        use_trailing_stop = trailing_stop.get("enabled", False)
        activation_percentage = trailing_stop.get("activation_pct", 1.0) / 100.0
        trail_percentage = trailing_stop.get("trail_pct", 0.5) / 100.0
        
        return {
            "position_size": position_size,
            "entry_price": current_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "leverage": leverage,
            "risk_amount": balance * risk_percentage,
            "use_trailing_stop": use_trailing_stop,
            "activation_percentage": activation_percentage,
            "trail_percentage": trail_percentage
        }

    def execute_trade(self, direction: str, risk_params: Dict) -> Optional[Dict]:
        """
        Execute a trade based on the analysis
        
        Args:
            direction: Trade direction ('buy' or 'sell')
            risk_params: Risk parameters
            
        Returns:
            Dict: Trade result or None if no trade was executed
        """
        # Skip if direction is 'none' or trading is paused
        if direction == "none" or self.trading_paused:
            return None
        
        # Check if we already have a position
        has_position = self.check_existing_position()
        
        # Skip if we already have a position in this direction
        if has_position:
            position = self.current_positions.get(self.symbol, {})
            position_side = position.get("side", "")
            
            if (direction == "buy" and position_side == "long") or (direction == "sell" and position_side == "short"):
                self.logger.info(f"Already have a {position_side} position, skipping {direction} trade")
                return None
            
            # Handle case where we want to reverse the position
            self.logger.info(f"Have a {position_side} position, but signal is {direction}. Considering position reversal.")
            
            # Check if position reversal is allowed
            if self.config.get("risk_management", {}).get("allow_position_reversal", False):
                # Close existing position first
                self.logger.info(f"Closing existing {position_side} position before opening {direction} position")
                close_result = self.close_position(position_side)
                
                if not close_result:
                    self.logger.warning("Failed to close existing position, cannot reverse")
                    return None
                
                # Wait a bit before opening the new position
                time.sleep(2)
                
                # Refetch market data
                self.update_candles()
                self.update_positions()
            else:
                self.logger.info("Position reversal not allowed in configuration")
                return None
        
        # Check maximum open positions
        max_positions = self.config.get("risk_management", {}).get("max_open_positions", 1)
        open_positions = len(self.current_positions)
        
        if open_positions >= max_positions:
            self.logger.info(f"Maximum open positions ({max_positions}) reached, skipping trade")
            return None
        
        # Prepare trade parameters
        size = risk_params.get("position_size", 0)
        
        if size <= 0:
            self.logger.warning(f"Invalid position size: {size}, skipping trade")
            return None
        
        price = risk_params.get("entry_price")
        sl_price = risk_params.get("stop_loss")
        tp_price = risk_params.get("take_profit")
        leverage = risk_params.get("leverage", 1.0)
        
        # Set trade parameters
        trade_params: TradeParams = {
            "symbol": self.symbol,
            "side": direction,
            "size": size,
            "price": price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "leverage": leverage,
            "reduce_only": False,
            "order_type": "market",
            "params": {}
        }
        
        # Execute trade
        try:
            # Set leverage if needed
            if leverage > 1 and self.market_info and self.market_info.get("is_contract", False):
                self.set_leverage(leverage)
            
            # Create trade
            if self.config.get("dry_run", True):
                # Simulate order in dry run mode
                self.logger.info(f"[DRY RUN] Creating {direction} order: {size} {self.symbol} @ {price}")
                order_result = self.simulate_order(trade_params)
            else:
                # Create real order
                self.logger.info(f"Creating {direction} order: {size} {self.symbol} @ {price}")
                order_result = self.create_order(trade_params)
            
            if not order_result:
                self.logger.warning("Failed to create order")
                return None
            
            # Update state
            self.update_positions()
            
            # Record trade in state
            trade_record = {
                "id": order_result.get("id", f"trade_{int(time.time() * 1000)}"),
                "symbol": self.symbol,
                "side": direction,
                "size": size,
                "entry_price": price,
                "timestamp": int(time.time() * 1000),
                "stop_loss": sl_price,
                "take_profit": tp_price,
                "leverage": leverage,
                "strategy": self.strategy_name,
                "order_type": "market",
                "status": "open"
            }
            
            # Add trade to state
            if "recent" not in self.state["trades"]:
                self.state["trades"]["recent"] = []
            
            self.state["trades"]["recent"].append(trade_record)
            self.state["trades"]["total"] += 1
            self.save_state()
            
            return trade_record
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            self.error_count += 1
            return None

    def check_existing_position(self) -> bool:
        """
        Check if we already have an open position for the symbol
        
        Returns:
            bool: True if position exists, False otherwise
        """
        return self.symbol in self.current_positions

    def set_leverage(self, leverage: float) -> bool:
        """
        Set leverage for the symbol
        
        Args:
            leverage: Leverage value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Skip in dry run mode
            if self.config.get("dry_run", True):
                self.logger.info(f"[DRY RUN] Setting leverage to {leverage}x for {self.symbol}")
                return True
            
            self.logger.info(f"Setting leverage to {leverage}x for {self.symbol}")
            
            result = retry_api_call(
                lambda: self.exchange.set_leverage(leverage, self.symbol)
            )
            
            self.logger.info(f"Leverage set to {leverage}x for {self.symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False

    def create_order(self, params: TradeParams) -> Optional[Dict]:
        """
        Create an order on the exchange
        
        Args:
            params: Trade parameters
            
        Returns:
            Dict: Order result or None if failed
        """
        try:
            # Prepare order parameters
            order_type = params.get("order_type", "market")
            side = params.get("side", "")
            amount = params.get("size", 0)
            price = params.get("price")
            
            # Convert side to exchange format
            if side == "buy":
                order_side = "buy"
            elif side == "sell":
                order_side = "sell"
            else:
                self.logger.error(f"Invalid side: {side}")
                return None
            
            # Prepare extra parameters
            extra_params = params.get("params", {})
            
            # Add stop loss and take profit if supported
            if self.config.get("risk_management", {}).get("use_sl_tp", True):
                sl_price = params.get("stop_loss")
                tp_price = params.get("take_profit")
                
                if self.exchange_id == "bybit":
                    # Bybit supports SL/TP in the same order
                    extra_params["stopLoss"] = sl_price
                    extra_params["takeProfit"] = tp_price
                    extra_params["reduce_only"] = False
            
            # Create the order
            order = retry_api_call(
                lambda: self.exchange.create_order(
                    symbol=self.symbol,
                    type=order_type,
                    side=order_side,
                    amount=amount,
                    price=price if order_type == "limit" else None,
                    params=extra_params
                )
            )
            
            self.logger.info(f"Order created: {order.get('id')} - {order_side} {amount} {self.symbol}")
            
            # If exchange doesn't support SL/TP in the same order, create separate orders
            if self.config.get("risk_management", {}).get("use_sl_tp", True) and self.exchange_id != "bybit":
                sl_price = params.get("stop_loss")
                tp_price = params.get("take_profit")
                
                # Create stop loss order
                if sl_price:
                    sl_side = "sell" if order_side == "buy" else "buy"
                    self.logger.info(f"Creating stop loss order: {sl_side} {amount} {self.symbol} @ {sl_price}")
                    
                    sl_order = retry_api_call(
                        lambda: self.exchange.create_order(
                            symbol=self.symbol,
                            type="stop",
                            side=sl_side,
                            amount=amount,
                            price=sl_price,
                            params={"stopPrice": sl_price, "reduce_only": True}
                        )
                    )
                    
                    self.logger.info(f"Stop loss order created: {sl_order.get('id')}")
                
                # Create take profit order
                if tp_price:
                    tp_side = "sell" if order_side == "buy" else "buy"
                    self.logger.info(f"Creating take profit order: {tp_side} {amount} {self.symbol} @ {tp_price}")
                    
                    tp_order = retry_api_call(
                        lambda: self.exchange.create_order(
                            symbol=self.symbol,
                            type="limit",
                            side=tp_side,
                            amount=amount,
                            price=tp_price,
                            params={"reduce_only": True}
                        )
                    )
                    
                    self.logger.info(f"Take profit order created: {tp_order.get('id')}")
            
            return order
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            self.error_count += 1
            return None

    def simulate_order(self, params: TradeParams) -> Dict:
        """
        Simulate an order in dry run mode
        
        Args:
            params: Trade parameters
            
        Returns:
            Dict: Simulated order result
        """
        # Create a simulated order response
        order_id = f"dryrun_{int(time.time() * 1000)}"
        
        order = {
            "id": order_id,
            "symbol": params["symbol"],
            "type": params["order_type"],
            "side": params["side"],
            "amount": params["size"],
            "price": params["price"],
            "status": "closed",
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().isoformat(),
            "fee": {
                "cost": params["size"] * params["price"] * 0.0005,
                "currency": "USDT"
            },
            "info": {
                "stop_loss": params["stop_loss"],
                "take_profit": params["take_profit"],
                "leverage": params["leverage"],
                "simulated": True
            }
        }
        
        # Simulate adding to the positions
        self.current_positions[params["symbol"]] = {
            "symbol": params["symbol"],
            "side": "long" if params["side"] == "buy" else "short",
            "contracts": params["size"],
            "entryPrice": params["price"],
            "leverage": params["leverage"],
            "liquidationPrice": 0.0,
            "unrealizedPnl": 0.0,
            "simulated": True
        }
        
        return order

    def close_position(self, position_side: str) -> Optional[Dict]:
        """
        Close an open position
        
        Args:
            position_side: Position side ('long' or 'short')
            
        Returns:
            Dict: Order result or None if failed
        """
        try:
            position = self.current_positions.get(self.symbol, {})
            if not position:
                self.logger.warning(f"No position found for {self.symbol}")
                return None
            
            size = position.get("contracts", 0)
            if size <= 0:
                self.logger.warning(f"Invalid position size: {size}")
                return None
            
            # Determine close side
            side = "sell" if position_side == "long" else "buy"
            
            # Create close order parameters
            close_params = {
                "symbol": self.symbol,
                "side": side,
                "size": size,
                "price": self.get_current_price(),
                "order_type": "market",
                "reduce_only": True,
                "params": {"reduce_only": True}
            }
            
            if self.config.get("dry_run", True):
                # Simulate close in dry run mode
                self.logger.info(f"[DRY RUN] Closing {position_side} position: {size} {self.symbol}")
                order_result = self.simulate_close(close_params)
            else:
                # Create real order
                self.logger.info(f"Closing {position_side} position: {size} {self.symbol}")
                order_result = self.create_order(close_params)
            
            if not order_result:
                self.logger.warning("Failed to close position")
                return None
            
            # Update positions and record in state
            self.update_positions()
            
            # Find the corresponding open trade record
            for trade in self.state["trades"].get("recent", []):
                if (trade.get("symbol") == self.symbol and 
                    trade.get("side") == "buy" and position_side == "long" or
                    trade.get("side") == "sell" and position_side == "short" and
                    trade.get("status") == "open"):
                    
                    # Update the trade record
                    entry_price = trade.get("entry_price", 0)
                    exit_price = self.get_current_price()
                    
                    # Calculate PnL
                    if position_side == "long":
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    
                    trade["exit_price"] = exit_price
                    trade["exit_time"] = int(time.time() * 1000)
                    trade["pnl"] = pnl_pct
                    trade["status"] = "closed"
                    
                    # Update win/loss counters
                    if pnl_pct > 0:
                        self.state["trades"]["wins"] += 1
                    else:
                        self.state["trades"]["losses"] += 1
                    
                    # Update strategy-specific stats
                    if self.strategy_name not in self.state["strategy_stats"]:
                        self.state["strategy_stats"][self.strategy_name] = {
                            "trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "pnl": 0.0
                        }
                    
                    self.state["strategy_stats"][self.strategy_name]["trades"] += 1
                    if pnl_pct > 0:
                        self.state["strategy_stats"][self.strategy_name]["wins"] += 1
                    else:
                        self.state["strategy_stats"][self.strategy_name]["losses"] += 1
                    
                    self.state["strategy_stats"][self.strategy_name]["pnl"] += pnl_pct
                    
                    # Save state
                    self.save_state()
                    
                    break
            
            return order_result
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            self.error_count += 1
            return None

    def simulate_close(self, params: Dict) -> Dict:
        """
        Simulate closing a position in dry run mode
        
        Args:
            params: Close parameters
            
        Returns:
            Dict: Simulated order result
        """
        # Create a simulated close order response
        order_id = f"dryrun_close_{int(time.time() * 1000)}"
        
        order = {
            "id": order_id,
            "symbol": params["symbol"],
            "type": params["order_type"],
            "side": params["side"],
            "amount": params["size"],
            "price": params["price"],
            "status": "closed",
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().isoformat(),
            "fee": {
                "cost": params["size"] * params["price"] * 0.0005,
                "currency": "USDT"
            },
            "info": {
                "reduce_only": True,
                "simulated": True
            }
        }
        
        # Remove from simulated positions
        if params["symbol"] in self.current_positions:
            del self.current_positions[params["symbol"]]
        
        return order

    def get_current_price(self) -> Optional[float]:
        """
        Get the current price for the symbol
        
        Returns:
            float: Current price or None if unavailable
        """
        try:
            if self.candles_df is not None and len(self.candles_df) > 0:
                # Use the last close price from candles
                return self.candles_df["close"].iloc[-1]
            
            # Fetch ticker as fallback
            ticker = retry_api_call(
                lambda: self.exchange.fetch_ticker(self.symbol)
            )
            
            if ticker and "last" in ticker:
                return ticker["last"]
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    def update_trailing_stops(self) -> None:
        """Update trailing stops for open positions"""
        try:
            # Skip if in dry run or no positions
            if self.config.get("dry_run", True) or not self.current_positions:
                return
            
            position = self.current_positions.get(self.symbol)
            if not position:
                return
            
            # Check if trailing stop is enabled
            trail_config = self.config.get("risk_management", {}).get("trailing_stop", {})
            if not trail_config.get("enabled", False):
                return
            
            # Get position details
            position_side = position.get("side", "")
            entry_price = position.get("entryPrice", 0)
            current_price = self.get_current_price()
            
            if not current_price or not entry_price:
                return
            
            # Calculate activation threshold
            activation_pct = trail_config.get("activation_pct", 1.0) / 100.0
            trail_pct = trail_config.get("trail_pct", 0.5) / 100.0
            
            # Check if position has reached activation threshold
            if position_side == "long":
                activation_price = entry_price * (1 + activation_pct)
                if current_price >= activation_price:
                    # Calculate new stop loss
                    new_stop = current_price * (1 - trail_pct)
                    self.logger.info(f"Updating trailing stop for {self.symbol}: {new_stop}")
                    self.update_stop_loss(new_stop)
            
            elif position_side == "short":
                activation_price = entry_price * (1 - activation_pct)
                if current_price <= activation_price:
                    # Calculate new stop loss
                    new_stop = current_price * (1 + trail_pct)
                    self.logger.info(f"Updating trailing stop for {self.symbol}: {new_stop}")
                    self.update_stop_loss(new_stop)
        
        except Exception as e:
            self.logger.error(f"Error updating trailing stops: {e}")

    def update_stop_loss(self, new_stop: float) -> bool:
        """
        Update stop loss for an open position
        
        Args:
            new_stop: New stop loss price
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Skip in dry run mode
            if self.config.get("dry_run", True):
                self.logger.info(f"[DRY RUN] Updating stop loss to {new_stop} for {self.symbol}")
                return True
            
            # Get position and open orders
            position = self.current_positions.get(self.symbol)
            if not position:
                return False
            
            # Fetch open orders
            open_orders = retry_api_call(
                lambda: self.exchange.fetch_open_orders(self.symbol)
            )
            
            # Find existing stop loss order
            sl_order = None
            for order in open_orders:
                if order.get("type") == "stop" or "stopPrice" in order.get("info", {}):
                    sl_order = order
                    break
            
            # Cancel existing stop loss if found
            if sl_order:
                self.logger.info(f"Canceling existing stop loss order: {sl_order.get('id')}")
                cancel_result = retry_api_call(
                    lambda: self.exchange.cancel_order(sl_order.get("id"), self.symbol)
                )
                
                if not cancel_result:
                    self.logger.warning("Failed to cancel existing stop loss order")
                    return False
            
            # Create new stop loss order
            position_side = position.get("side", "")
            size = position.get("contracts", 0)
            
            # Determine side for stop loss
            sl_side = "sell" if position_side == "long" else "buy"
            
            self.logger.info(f"Creating new stop loss order: {sl_side} {size} {self.symbol} @ {new_stop}")
            
            sl_order = retry_api_call(
                lambda: self.exchange.create_order(
                    symbol=self.symbol,
                    type="stop",
                    side=sl_side,
                    amount=size,
                    price=new_stop,
                    params={"stopPrice": new_stop, "reduce_only": True}
                )
            )
            
            if not sl_order:
                self.logger.warning("Failed to create new stop loss order")
                return False
            
            self.logger.info(f"New stop loss order created: {sl_order.get('id')}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")
            return False

    def handle_exit_signals(self, analysis: Dict) -> None:
        """
        Handle exit signals from the strategy
        
        Args:
            analysis: Analysis results containing signal information
        """
        try:
            # Skip if no positions or exit_signal not present
            if not self.current_positions or "exit_signal" not in analysis:
                return
            
            position = self.current_positions.get(self.symbol)
            if not position:
                return
            
            exit_signal = analysis.get("exit_signal", False)
            if not exit_signal:
                return
            
            # Get position details
            position_side = position.get("side", "")
            self.logger.info(f"Exit signal received for {position_side} position in {self.symbol}")
            
            # Close the position
            self.close_position(position_side)
        
        except Exception as e:
            self.logger.error(f"Error handling exit signals: {e}")

    def run(self, stop_event: Optional[Event] = None) -> None:
        """
        Run the trading bot main loop
        
        Args:
            stop_event: Event to signal stopping the bot
        """
        local_stop_event = stop_event or Event()
        
        self.logger.info(f"Starting trading bot main loop for {self.symbol} on {self.exchange_id}")
        self.logger.info(f"Strategy: {self.strategy_name}")
        
        # Track loop iterations and errors
        iterations = 0
        consecutive_errors = 0
        
        try:
            while not local_stop_event.is_set():
                try:
                    # Increment iteration counter
                    iterations += 1
                    
                    # Fetch updated market data
                    self.update_candles()
                    
                    # Update positions
                    self.update_positions()
                    
                    # Check if we should update balance (every 10 iterations)
                    if iterations % 10 == 0:
                        self.update_balance()
                    
                    # Analyze market
                    analysis = self.analyze_market()
                    
                    # Log analysis results
                    signal_strength = analysis.get("signal_strength", 0)
                    direction = analysis.get("direction", "none")
                    
                    self.logger.info(
                        f"Analysis results for {self.symbol}: "
                        f"Signal={signal_strength:.2f}, Direction={direction}"
                    )
                    
                    # Handle exit signals first
                    self.handle_exit_signals(analysis)
                    
                    # Update trailing stops for open positions
                    self.update_trailing_stops()
                    
                    # Check signal strength threshold for entry
                    entry_threshold = self.config.get("strategy", {}).get("entry_threshold", 0.5)
                    
                    if abs(signal_strength) >= entry_threshold and direction != "none":
                        # Execute trade based on signal
                        risk_params = analysis.get("risk_params", {})
                        trade_result = self.execute_trade(direction, risk_params)
                        
                        if trade_result:
                            self.logger.info(f"Trade executed: {direction} {trade_result.get('size')} {self.symbol}")
                    
                    # Reset consecutive error counter on successful iteration
                    consecutive_errors = 0
                    
                    # Save state periodically
                    if iterations % 5 == 0:
                        self.save_state()
                    
                    # Calculate sleep time based on configuration
                    loop_interval = self.config.get("loop_interval_seconds", 15)
                    
                    # Sleep but check for stop event periodically
                    for _ in range(loop_interval):
                        if local_stop_event.is_set():
                            break
                        time.sleep(1)
                
                except Exception as e:
                    # Increment error counter
                    consecutive_errors += 1
                    self.error_count += 1
                    
                    # Log error
                    self.logger.error(f"Error in main loop (iteration {iterations}): {e}")
                    
                    # Record error in state
                    error_info = {
                        "timestamp": int(time.time() * 1000),
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
                    self.state["errors"]["last_error"] = str(e)
                    self.state["errors"]["last_error_time"] = int(time.time() * 1000)
                    self.state["errors"]["error_count"] += 1
                    
                    if "recent_errors" not in self.state["errors"]:
                        self.state["errors"]["recent_errors"] = []
                    
                    # Keep only the last 10 errors
                    self.state["errors"]["recent_errors"].append(error_info)
                    self.state["errors"]["recent_errors"] = self.state["errors"]["recent_errors"][-10:]
                    
                    self.save_state()
                    
                    # Stop bot if too many consecutive errors
                    max_consecutive_errors = self.config.get("advanced", {}).get("max_consecutive_errors", 5)
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping bot")
                        break
                    
                    # Sleep before retry
                    time.sleep(5)
        
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        
        finally:
            # Final cleanup
            self.logger.info("Saving final state")
            self.state["active"] = False
            self.save_state()
            
            self.logger.info("Bot stopped")

    def run_backtest(self) -> None:
        """Run backtest using the current configuration"""
        self.logger.info("Backtesting not implemented in base class")
        # This should be implemented in a separate backtesting module

    def validate_config(self) -> Dict:
        """
        Validate the bot configuration
        
        Returns:
            Dict: Validation result
        """
        errors = []
        
        # Basic configuration checks
        if not self.config.get("exchange"):
            errors.append("Exchange not specified")
        
        if not self.config.get("symbol"):
            errors.append("Trading symbol not specified")
        
        if not self.config.get("timeframe"):
            errors.append("Timeframe not specified")
        
        # Strategy checks
        if not self.config.get("strategy", {}).get("active"):
            errors.append("Active strategy not specified")
        
        # Risk management checks
        risk_config = self.config.get("risk_management", {})
        if risk_config.get("position_size_pct", 0) <= 0:
            errors.append("Position size percentage must be greater than 0")
        
        if risk_config.get("max_open_positions", 0) <= 0:
            errors.append("Maximum open positions must be greater than 0")
        
        # Create validation result
        if errors:
            return {
                "valid": False,
                "errors": errors
            }
        else:
            # Return successful validation with details
            return {
                "valid": True,
                "details": {
                    "exchange": self.config.get("exchange"),
                    "symbol": self.config.get("symbol"),
                    "timeframe": self.config.get("timeframe"),
                    "strategy": self.config.get("strategy", {}).get("active"),
                    "max_positions": risk_config.get("max_open_positions"),
                    "position_size": f"{risk_config.get('position_size_pct')}%",
                    "mode": "Dry run" if self.config.get("dry_run", True) else "Live trading"
                }
            }

    def set_symbol(self, symbol: str) -> None:
        """
        Set trading symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
        """
        self.symbol = symbol
        self.config["symbol"] = symbol
        self.logger.info(f"Symbol set to {symbol}")

    def set_timeframe(self, timeframe: str) -> None:
        """
        Set trading timeframe
        
        Args:
            timeframe: Trading timeframe (e.g., '15m')
        """
        self.timeframe = timeframe
        self.config["timeframe"] = timeframe
        self.logger.info(f"Timeframe set to {timeframe}")

    def set_exchange(self, exchange: str) -> None:
        """
        Set exchange
        
        Args:
            exchange: Exchange name (e.g., 'bybit')
        """
        self.exchange_id = exchange
        self.config["exchange"] = exchange
        self.logger.info(f"Exchange set to {exchange}")

    def set_strategy(self, strategy: str) -> None:
        """
        Set active strategy
        
        Args:
            strategy: Strategy name (e.g., 'ehlers_supertrend')
        """
        self.strategy_name = strategy
        self.config["strategy"] = {"active": strategy}
        self.logger.info(f"Strategy set to {strategy}")

    def fetch_candles(self, limit: int = 100) -> List:
        """
        Fetch OHLCV candles for the configured symbol and timeframe
        
        Args:
            limit: Number of candles to fetch
            
        Returns:
            List: Candles data
        """
        try:
            self.logger.info(f"Fetching {limit} candles for {self.symbol} ({self.timeframe})")
            candles = retry_api_call(
                lambda: self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    limit=limit
                )
            )
            
            return candles
        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            return []

    def fetch_ticker(self) -> Dict:
        """
        Fetch ticker data for the symbol
        
        Returns:
            Dict: Ticker data
        """
        try:
            ticker = retry_api_call(
                lambda: self.exchange.fetch_ticker(self.symbol)
            )
            
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            return {}