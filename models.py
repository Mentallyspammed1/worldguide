"""
Database Models Module

This module defines the database models for the Flask application.
"""

from datetime import datetime
import logging

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

from app import db

logger = logging.getLogger(__name__)

class User(UserMixin, db.Model):
    """User model for authentication and personalization"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Relationships
    api_keys = relationship('ApiKey', back_populates='user', cascade='all, delete-orphan')
    settings = relationship('UserSetting', back_populates='user', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)


class ApiKey(db.Model):
    """API key model for exchange connections"""
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    exchange = Column(String(64), nullable=False)
    name = Column(String(64))
    api_key = Column(String(128), nullable=False)
    api_secret = Column(String(256), nullable=False)
    testnet = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    
    # Relationships
    user = relationship('User', back_populates='api_keys')
    
    def __repr__(self):
        return f'<ApiKey {self.exchange}:{self.name}>'


class UserSetting(db.Model):
    """User settings model"""
    __tablename__ = 'user_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    category = Column(String(64), nullable=False)
    name = Column(String(64), nullable=False)
    value = Column(Text)
    
    # Relationships
    user = relationship('User', back_populates='settings')
    
    def __repr__(self):
        return f'<UserSetting {self.category}.{self.name}>'


class TradeHistory(db.Model):
    """Trading history model"""
    __tablename__ = 'trade_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    exchange = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    order_id = Column(String(64))
    side = Column(String(16), nullable=False)  # 'buy', 'sell', 'long', 'short'
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    pnl = Column(Float)
    fee = Column(Float)
    strategy = Column(String(64))
    status = Column(String(32), default='open')  # 'open', 'closed', 'cancelled'
    note = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<TradeHistory {self.symbol} {self.side} {self.size} @ {self.entry_price}>'


class BotConfig(db.Model):
    """Trading configuration model"""
    __tablename__ = 'bot_configs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String(64), nullable=False)
    exchange = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    timeframe = Column(String(16), nullable=False)
    strategy = Column(String(64), nullable=False)
    position_size_pct = Column(Float, default=1.0)
    max_open_positions = Column(Integer, default=1)
    leverage = Column(Float, default=1.0)
    is_active = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    config_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<BotConfig {self.name} {self.exchange}:{self.symbol}>'


class Position(db.Model):
    """Position model for tracking open positions"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    exchange = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(16), nullable=False)  # 'long' or 'short'
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    leverage = Column(Float, default=1.0)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Position {self.symbol} {self.side} {self.size}>'
