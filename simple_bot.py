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
    Simplified trading bot class for UI testing
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
        """Initialize the mock trading bot"""
        self.logger = logging.getLogger("trading_bot")
        self.config_file = config_file
        self.config = self._get_default_config()
        self.state = self._get_default_state()
        self.exchange_id = "bybit"
        self.exchange = None
        self.symbol = "BTC/USDT"
        self.timeframe = "15m"
        self.candles_df = self._generate_mock_candles()
        self.current_positions = {}
        
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
        """Set up exchange connection (mock implementation)"""
        self.logger.info(f"Loading markets for {self.exchange_id}")
        # In a real implementation, this would connect to the exchange
        return None
    
    def initialize(self) -> None:
        """Initialize the bot (mock implementation)"""
        return
    
    def get_balance(self) -> Dict:
        """Get account balance (mock implementation)"""
        return self.state["balance"]
    
    def get_ticker(self) -> Dict:
        """Get ticker data (mock implementation)"""
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
        """Update candles data (mock implementation)"""
        return self.candles_df
    
    def update_positions(self) -> Dict:
        """Update positions (mock implementation)"""
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
    
    def close_position(self, position_side: str) -> Optional[Dict]:
        """Close position (mock implementation)"""
        self.logger.info(f"Closing {position_side} position for {self.symbol}")
        return {"status": "closed"}


# Alias MockBot to TradingBot for compatibility
MockBot = TradingBot