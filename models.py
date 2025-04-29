"""
Database Models Module

This module defines the SQLAlchemy models for the database.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

from app import db


class User(UserMixin, db.Model):
    """User model for authentication and account management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    configs = db.relationship('TradingConfig', backref='user', lazy='dynamic')
    positions = db.relationship('Position', backref='user', lazy='dynamic')
    trades = db.relationship('TradeHistory', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)


class TradingConfig(db.Model):
    """Trading configuration model for storing bot settings"""
    __tablename__ = 'trading_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(64), nullable=False)
    exchange = db.Column(db.String(64), nullable=False, default='bybit')
    symbol = db.Column(db.String(32), nullable=False, default='BTC/USDT')
    timeframe = db.Column(db.String(8), nullable=False, default='15m')
    
    # Strategy configurations stored as JSON
    strategy = db.Column(db.JSON, nullable=False, default={})
    risk_management = db.Column(db.JSON, nullable=False, default={})
    indicators = db.Column(db.JSON, nullable=False, default={})
    
    is_active = db.Column(db.Boolean, default=False)
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<TradingConfig {self.name} for {self.exchange}:{self.symbol}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy': self.strategy,
            'risk_management': self.risk_management,
            'indicators': self.indicators,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Position(db.Model):
    """Position model for tracking open trading positions"""
    __tablename__ = 'positions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exchange = db.Column(db.String(64), nullable=False)
    symbol = db.Column(db.String(32), nullable=False)
    side = db.Column(db.String(8), nullable=False)  # 'long' or 'short'
    size = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float)
    
    # Risk parameters
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    leverage = db.Column(db.Float, default=1.0)
    
    # Status
    is_open = db.Column(db.Boolean, default=True)
    unrealized_pnl = db.Column(db.Float, default=0.0)
    unrealized_pnl_percentage = db.Column(db.Float, default=0.0)
    
    # Strategy info
    strategy = db.Column(db.String(64))
    entry_signal_strength = db.Column(db.Float)
    entry_indicators = db.Column(db.JSON)
    
    # Timestamps
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Position {self.side} {self.size} {self.symbol} @ {self.entry_price}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'leverage': self.leverage,
            'is_open': self.is_open,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percentage': self.unrealized_pnl_percentage,
            'strategy': self.strategy,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class TradeHistory(db.Model):
    """Trade history model for tracking completed trades"""
    __tablename__ = 'trade_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exchange = db.Column(db.String(64), nullable=False)
    symbol = db.Column(db.String(32), nullable=False)
    side = db.Column(db.String(8), nullable=False)  # 'long' or 'short'
    size = db.Column(db.Float, nullable=False)
    
    # Trade details
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float, nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime, nullable=False)
    
    # Performance metrics
    pnl = db.Column(db.Float, nullable=False)  # Absolute profit/loss
    pnl_percentage = db.Column(db.Float, nullable=False)  # Percentage profit/loss
    fees = db.Column(db.Float, default=0.0)
    
    # Risk parameters used
    leverage = db.Column(db.Float, default=1.0)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    risk_reward_ratio = db.Column(db.Float)
    
    # Strategy info
    strategy = db.Column(db.String(64))
    exit_reason = db.Column(db.String(64))  # 'take_profit', 'stop_loss', 'manual', etc.
    notes = db.Column(db.Text)
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Trade {self.side} {self.symbol} {self.pnl_percentage:.2f}%>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'fees': self.fees,
            'leverage': self.leverage,
            'strategy': self.strategy,
            'exit_reason': self.exit_reason,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class ExchangeCredential(db.Model):
    """Exchange API credentials model, encrypted in the database"""
    __tablename__ = 'exchange_credentials'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exchange = db.Column(db.String(64), nullable=False)
    api_key = db.Column(db.String(256), nullable=False)
    api_secret = db.Column(db.String(256), nullable=False)
    is_testnet = db.Column(db.Boolean, default=True)
    note = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<ExchangeCredential {self.exchange} for User {self.user_id}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization (without sensitive data)"""
        return {
            'id': self.id,
            'exchange': self.exchange,
            'is_testnet': self.is_testnet,
            'note': self.note,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


class BotLog(db.Model):
    """Log entry model for tracking bot activity"""
    __tablename__ = 'bot_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    level = db.Column(db.String(16), nullable=False)  # 'INFO', 'WARNING', 'ERROR', etc.
    component = db.Column(db.String(64), nullable=False)  # 'trading_bot', 'web_interface', etc.
    message = db.Column(db.Text, nullable=False)
    details = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<BotLog {self.level} {self.component}: {self.message[:30]}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'level': self.level,
            'component': self.component,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }