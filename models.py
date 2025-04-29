"""
Database Models for Trading Bot

This module defines the SQLAlchemy ORM models for the trading bot, including:
- User: For authentication and user management
- Trade: For recording trade history
- Position: For tracking current positions
- Strategy: For storing strategy configurations
- ExchangeAccount: For managing exchange API connections
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app import db


class User(UserMixin, db.Model):
    """User model for authentication and personalization"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    exchange_accounts = db.relationship('ExchangeAccount', backref='user', lazy='dynamic')
    trades = db.relationship('Trade', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class ExchangeAccount(db.Model):
    """Model for storing exchange API credentials and settings"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(64), nullable=False)
    exchange = db.Column(db.String(64), nullable=False)  # e.g., 'bybit', 'binance'
    api_key_encrypted = db.Column(db.Text)
    api_secret_encrypted = db.Column(db.Text)
    is_testnet = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    positions = db.relationship('Position', backref='exchange_account', lazy='dynamic')
    trades = db.relationship('Trade', backref='exchange_account', lazy='dynamic')
    
    def __repr__(self):
        return f'<ExchangeAccount {self.name} ({self.exchange})>'


class Strategy(db.Model):
    """Model for storing strategy configurations"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    type = db.Column(db.String(64), nullable=False)  # e.g., 'ehlers_supertrend', 'momentum_divergence'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    config = db.Column(db.JSON)  # Strategy-specific configuration
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    trades = db.relationship('Trade', backref='strategy', lazy='dynamic')
    positions = db.relationship('Position', backref='strategy', lazy='dynamic')
    
    def __repr__(self):
        return f'<Strategy {self.name} ({self.type})>'


class Position(db.Model):
    """Model for tracking open positions"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    exchange_account_id = db.Column(db.Integer, db.ForeignKey('exchange_account.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=True)
    symbol = db.Column(db.String(32), nullable=False)
    side = db.Column(db.String(16), nullable=False)  # 'long' or 'short'
    size = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float)
    leverage = db.Column(db.Float, default=1.0)
    liquidation_price = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    trailing_stop = db.Column(db.Boolean, default=False)
    trailing_stop_activation = db.Column(db.Float)
    trailing_stop_callback = db.Column(db.Float)
    unrealized_pnl = db.Column(db.Float, default=0.0)
    unrealized_pnl_pct = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Additional fields for exchange-specific info
    exchange_position_id = db.Column(db.String(64))
    margin_mode = db.Column(db.String(16))  # 'isolated' or 'cross'
    isolated_margin = db.Column(db.Float)
    
    def __repr__(self):
        return f'<Position {self.symbol} {self.side} {self.size}>'


class Trade(db.Model):
    """Model for recording trade history"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    exchange_account_id = db.Column(db.Integer, db.ForeignKey('exchange_account.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=True)
    symbol = db.Column(db.String(32), nullable=False)
    side = db.Column(db.String(16), nullable=False)  # 'long' or 'short'
    size = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    entry_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime)
    pnl = db.Column(db.Float)
    pnl_pct = db.Column(db.Float)
    fees = db.Column(db.Float, default=0.0)
    exit_reason = db.Column(db.String(32))  # 'take_profit', 'stop_loss', 'manual', etc.
    
    # Exchange-specific info
    exchange_order_id_entry = db.Column(db.String(64))
    exchange_order_id_exit = db.Column(db.String(64))
    
    # Additional metadata
    leverage = db.Column(db.Float, default=1.0)
    signal_strength = db.Column(db.Float)
    market_conditions = db.Column(db.JSON)
    notes = db.Column(db.Text)
    
    def __repr__(self):
        return f'<Trade {self.symbol} {self.side} {self.pnl_pct}%>'


class Setting(db.Model):
    """Model for application settings"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Null for global settings
    key = db.Column(db.String(64), nullable=False)
    value = db.Column(db.Text)
    value_type = db.Column(db.String(16), default='string')  # 'string', 'int', 'float', 'boolean', 'json'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'key', name='_user_key_uc'),
    )
    
    def __repr__(self):
        return f'<Setting {self.key}>'


class MarketData(db.Model):
    """Model for caching market data"""
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), nullable=False)
    timeframe = db.Column(db.String(8), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    
    # Technical indicators (optional)
    rsi = db.Column(db.Float)
    macd = db.Column(db.Float)
    macd_signal = db.Column(db.Float)
    macd_histogram = db.Column(db.Float)
    bb_upper = db.Column(db.Float)
    bb_middle = db.Column(db.Float)
    bb_lower = db.Column(db.Float)
    ema_fast = db.Column(db.Float)
    ema_slow = db.Column(db.Float)
    atr = db.Column(db.Float)
    
    __table_args__ = (
        db.UniqueConstraint('symbol', 'timeframe', 'timestamp', name='_market_data_uc'),
        db.Index('ix_market_data_symbol_timeframe', 'symbol', 'timeframe'),
        db.Index('ix_market_data_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f'<MarketData {self.symbol} {self.timeframe} {self.timestamp}>'