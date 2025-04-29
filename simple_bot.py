"""
Simple Bot Module

This is a simplified version of the trading bot for web UI testing.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, TypedDict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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
    Simplified trading bot class for UI testing with real API support
    """
    def __init__(
        self,
        config_file: str = "config.json",
        validate_only: bool = False,
        exchange: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        is_testnet: bool = True
    ):
        """Initialize the trading bot"""
        self.logger = logging.getLogger("trading_bot")
        self.config_file = config_file
        self.config = self._get_default_config()
        self.state = self._get_default_state()
        self.exchange_id = exchange or "bybit"
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_testnet = is_testnet
        self.exchange = None
        self.symbol = "BTC/USDT"
        self.timeframe = "15m"
        self.candles_df = self._generate_mock_candles()
        self.current_positions = {}
        
        # Try to initialize the exchange if API credentials are provided
        if api_key and api_secret:
            try:
                self.setup_exchange()
            except Exception as e:
                self.logger.error(f"Failed to initialize exchange: {e}")
                # Fall back to mock mode
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "exchange": "bybit",
            "symbol": "BTC/USDT",
            "timeframe": "15m",
            "dry_run": True,
            "strategy": {
                "active": "ehlers_supertrend",
                "params": {}
            },
            "risk_management": {
                "position_size_pct": 1.0,
                "max_open_positions": 3,
                "use_sl_tp": True,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0
            }
        }
        
    def _get_default_state(self) -> Dict:
        """Get default state"""
        return {
            "active": False,
            "positions": {},
            "balance": {"total": 10000, "free": 10000, "used": 0},
            "performance": {"pnl_percentage": 0.0, "drawdown_max": 0.0},
            "trades": {"total": 0, "wins": 0, "recent": []}
        }
        
    def _generate_mock_candles(self) -> pd.DataFrame:
        """Generate mock candles for UI testing"""
        # Create sample candles dataframe for the UI
        now = pd.Timestamp.now()
        periods = 100
        timestamps = pd.date_range(end=now, periods=periods, freq='15min')
        
        # Generate some random but plausible price data
        base_price = 50000
        price_data = []
        for i in range(periods):
            change = np.random.normal(0, 1) * 100  # Random price movement
            price = base_price + change
            base_price = price  # Use as new base for next iteration
            
            # Create OHLCV candle with some randomness
            open_price = price - np.random.random() * 50
            close_price = price + np.random.random() * 50
            high_price = max(open_price, close_price) + np.random.random() * 50
            low_price = min(open_price, close_price) - np.random.random() * 50
            volume = np.random.random() * 100 + 10
            
            price_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame with timestamps as index
        return pd.DataFrame(price_data, index=timestamps)
        
    def load_config(self) -> Dict:
        """Load configuration from file (mock implementation)"""
        self.logger.info(f"Loaded configuration from {self.config_file}")
        return self.config
    
    def setup_exchange(self) -> None:
        """Set up real exchange connection using CCXT"""
        import ccxt

        self.logger.info(f"Setting up connection to {self.exchange_id}")
        
        try:
            # Exchange connection parameters
            exchange_params = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            }
            
            # Add testnet configuration if needed
            if self.is_testnet:
                if self.exchange_id == 'bybit':
                    exchange_params['options'] = {
                        'testnet': True  # Use Bybit testnet
                    }
            
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(exchange_params)
            
            # Load markets
            self.logger.info(f"Loading markets for {self.exchange_id}")
            self.exchange.load_markets()
            
            # Update config
            self.config['dry_run'] = False
            self.logger.info(f"Successfully connected to {self.exchange_id}")
            
            return self.exchange
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            self.exchange = None
            return None
    
    def initialize(self) -> None:
        """Initialize the bot with current configuration"""
        try:
            # Setup exchange connection if not already connected
            if self.api_key and self.api_secret and not self.exchange:
                self.setup_exchange()
            
            self.logger.info("Bot initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing bot: {e}")
            return False
            
    @classmethod
    def validate_api_key(cls, exchange: str, api_key: str, api_secret: str, testnet: bool = True) -> Dict:
        """Validate API key by attempting to connect to the exchange"""
        import ccxt
        
        try:
            # Exchange connection parameters
            exchange_params = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            }
            
            # Add testnet configuration if needed
            if testnet:
                if exchange == 'bybit':
                    exchange_params['options'] = {
                        'testnet': True  # Use Bybit testnet
                    }
            
            # Create exchange instance
            exchange_class = getattr(ccxt, exchange)
            exchange_instance = exchange_class(exchange_params)
            
            # Test connection by fetching balance
            balance = exchange_instance.fetch_balance()
            
            return {
                'valid': True,
                'message': 'API key validation successful',
                'balance': balance
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'API key validation failed: {e}',
                'error': str(e)
            }
    
    def get_balance(self) -> Dict:
        """Get account balance from exchange or mock data"""
        if self.exchange:
            try:
                # Get real balance from exchange
                balance_data = self.exchange.fetch_balance()
                
                # Extract USDT balance
                if 'USDT' in balance_data['total']:
                    total = balance_data['total']['USDT']
                    free = balance_data['free']['USDT']
                    used = balance_data['used']['USDT']
                else:
                    # Fall back to first available currency
                    currencies = list(balance_data['total'].keys())
                    if currencies:
                        currency = currencies[0]
                        total = balance_data['total'][currency]
                        free = balance_data['free'][currency]
                        used = balance_data['used'][currency]
                    else:
                        # No balances found
                        return self.state["balance"]
                
                # Update state
                self.state["balance"] = {
                    "total": total,
                    "free": free,
                    "used": used
                }
                
                return self.state["balance"]
            except Exception as e:
                self.logger.error(f"Error fetching balance: {e}")
                return self.state["balance"]
        else:
            # Return mock balance
            return self.state["balance"]
    
    def get_ticker(self) -> Dict:
        """Get ticker data from exchange or mock data"""
        if self.exchange:
            try:
                # Get real ticker from exchange
                ticker_data = self.exchange.fetch_ticker(self.symbol)
                
                # Format ticker data
                return {
                    "symbol": self.symbol,
                    "last": ticker_data['last'],
                    "bid": ticker_data['bid'],
                    "ask": ticker_data['ask'],
                    "high": ticker_data['high'],
                    "low": ticker_data['low'],
                    "volume": ticker_data['volume'],
                    "percentage": ticker_data['percentage'] if 'percentage' in ticker_data else 0
                }
            except Exception as e:
                self.logger.error(f"Error fetching ticker: {e}")
                # Fall back to mock data
                return {
                    "symbol": self.symbol,
                    "last": 50000,
                    "bid": 49990,
                    "ask": 50010,
                    "high": 51000,
                    "low": 49000,
                    "volume": 1000.0,
                    "percentage": 0.5
                }
        else:
            # Return mock ticker
            return {
                "symbol": self.symbol,
                "last": 50000,
                "bid": 49990,
                "ask": 50010,
                "high": 51000,
                "low": 49000,
                "volume": 1000.0,
                "percentage": 0.5
            }
    
    def update_candles(self) -> pd.DataFrame:
        """Update candles data from exchange or use mock data"""
        if self.exchange:
            try:
                # Convert timeframe from '15m' format to '15min' for pandas
                timeframe_str = self.timeframe
                if timeframe_str.endswith('m'):
                    timeframe_str = timeframe_str.replace('m', 'min')
                
                # Get OHLCV data from exchange
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
                
                # Convert to pandas dataframe
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Update candles
                self.candles_df = df
                self.logger.info(f"Updated candles for {self.symbol} ({self.timeframe})")
                
                return self.candles_df
            except Exception as e:
                self.logger.error(f"Error updating candles: {e}")
                return self.candles_df
        else:
            # Return mock candles
            return self.candles_df
    
    def update_positions(self) -> Dict:
        """Update positions from exchange or use mock data"""
        if self.exchange:
            try:
                # Get positions from exchange
                positions = self.exchange.fetch_positions([self.symbol]) if hasattr(self.exchange, 'fetch_positions') else []
                
                # Process positions
                self.current_positions = {}
                
                for pos in positions:
                    side = pos['side'].lower()
                    size = float(pos['size']) if 'size' in pos else float(pos['contracts']) if 'contracts' in pos else 0.0
                    
                    # Skip zero-sized positions
                    if size <= 0:
                        continue
                    
                    # Add to positions dict
                    self.current_positions[side] = {
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': float(pos['entryPrice']) if 'entryPrice' in pos else 0.0,
                        'liquidation_price': float(pos['liquidationPrice']) if 'liquidationPrice' in pos else 0.0,
                        'margin': float(pos['initialMargin']) if 'initialMargin' in pos else 0.0,
                        'pnl': float(pos['unrealizedPnl']) if 'unrealizedPnl' in pos else 0.0,
                        'leverage': float(pos['leverage']) if 'leverage' in pos else 1.0
                    }
                
                return self.current_positions
            except Exception as e:
                self.logger.error(f"Error updating positions: {e}")
                return self.current_positions
        else:
            # Return mock positions
            return self.current_positions
    
    def start(self) -> bool:
        """Start the bot (mock implementation)"""
        self.state["active"] = True
        return True
    
    def stop(self) -> bool:
        """Stop the bot (mock implementation)"""
        self.state["active"] = False
        return True
    
    def analyze_market(self) -> Dict:
        """Analyze market (mock implementation)"""
        return {
            "timestamp": int(time.time() * 1000),
            "symbol": self.symbol,
            "strategy": self.config["strategy"]["active"],
            "signal_strength": 0,
            "direction": "none"
        }
    
    def close_position(self, position_side: str, symbol: str = None) -> Optional[Dict]:
        """Close position on exchange or mock close"""
        target_symbol = symbol or self.symbol
        self.logger.info(f"Closing {position_side} position for {target_symbol}")
        
        if self.exchange:
            try:
                # Get position details
                position = self.current_positions.get(position_side.lower())
                if not position:
                    self.logger.warning(f"No {position_side} position found for {target_symbol}")
                    return {"status": "error", "message": f"No {position_side} position found"}
                
                # Prepare order parameters
                # For Bybit and similar exchanges, we use a market order with reduce_only=True
                order_params = {
                    "symbol": target_symbol,
                    "type": "market",
                    "side": "buy" if position_side.lower() == "short" else "sell",  # Opposite side to close
                    "amount": position['size'],
                    "params": {
                        "reduce_only": True
                    }
                }
                
                # Execute the order
                self.logger.info(f"Executing close order: {order_params}")
                result = self.exchange.create_order(**order_params)
                
                # Update positions
                self.update_positions()
                
                return {
                    "status": "closed",
                    "order_id": result.get('id'),
                    "symbol": target_symbol,
                    "side": position_side,
                    "size": position['size'],
                    "price": result.get('price') or result.get('average')
                }
            except Exception as e:
                self.logger.error(f"Error closing position: {e}")
                return {"status": "error", "message": str(e)}
        else:
            # Mock implementation
            return {"status": "closed"}


# Alias MockBot to TradingBot for compatibility
MockBot = TradingBot