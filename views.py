"""
Views Module

This module defines the routes and view functions for the Flask application.
"""

import json
import os
import time
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional, Union, Any

import ccxt
import pandas as pd
from flask import (
    jsonify, redirect, render_template, request, session, url_for, flash
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from app import app, db
from models import User, TradingConfig, TradeHistory, Position
from trading_bot import TradingBot

# Configure logger
logger = logging.getLogger("views")

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Global bot instance
bot = None

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(int(user_id))


def initialize_bot():
    """Initialize the trading bot with configuration"""
    global bot
    
    try:
        if bot is None:
            # Create new bot instance
            config_path = app.config.get("CONFIG_PATH", "config.json")
            bot = TradingBot(config_file=config_path)
            logger.info("Trading bot initialized")
    except Exception as e:
        logger.error(f"Error initializing trading bot: {e}")
        bot = None


def get_bot_instance():
    """Get the current bot instance or initialize a new one"""
    global bot
    
    if bot is None:
        initialize_bot()
    
    return bot


@app.route('/')
def index():
    """Render the index page"""
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    # Initialize bot if needed
    bot = get_bot_instance()
    
    # Get bot state and configuration
    state = {}
    config = {}
    
    if bot:
        state = bot.state
        config = bot.config
    
    # Get trades from database
    trades = []
    try:
        recent_trades = TradeHistory.query.order_by(TradeHistory.timestamp.desc()).limit(10).all()
        for trade in recent_trades:
            trades.append({
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'time': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            })
    except SQLAlchemyError as e:
        logger.error(f"Database error when fetching trades: {e}")
    
    # Prepare template variables
    context = {
        'state': state,
        'config': config,
        'trades': trades,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('dashboard.html', **context)


@app.route('/settings')
def settings():
    """Render the settings page"""
    # Get bot instance and config
    bot = get_bot_instance()
    config = bot.config if bot else {}
    
    return render_template('settings.html', config=config)


@app.route('/accounts')
def accounts():
    """Render the accounts page"""
    # Get bot instance
    bot = get_bot_instance()
    
    # Get exchange accounts and balances
    accounts = []
    if bot:
        try:
            # Get balance from bot
            balance = bot.get_balance()
            
            # Add to accounts list
            accounts.append({
                'exchange': bot.exchange_id,
                'name': bot.exchange_id.capitalize(),
                'balance': balance,
                'connected': bot.exchange is not None
            })
        except Exception as e:
            logger.error(f"Error fetching account information: {e}")
    
    return render_template('accounts.html', accounts=accounts)


@app.route('/strategies')
def strategies():
    """Render the strategies page"""
    # Get bot instance
    bot = get_bot_instance()
    
    # Get strategies configuration
    strategies = []
    if bot:
        # Get strategy config
        strategy_config = bot.config.get('strategy', {})
        active_strategy = strategy_config.get('active', 'ehlers_supertrend')
        
        # Available strategies
        available_strategies = [
            {
                'id': 'ehlers_supertrend',
                'name': 'Ehlers Supertrend',
                'description': 'Trend-following strategy based on John Ehlers\' Supertrend indicator',
                'active': active_strategy == 'ehlers_supertrend'
            },
            {
                'id': 'momentum_divergence',
                'name': 'Momentum Divergence',
                'description': 'Reversal strategy using RSI/MACD divergence with volatility filters',
                'active': active_strategy == 'momentum_divergence'
            },
            {
                'id': 'multi_timeframe_trend',
                'name': 'Multi-Timeframe Trend',
                'description': 'Combines multiple timeframes for stronger trend signals',
                'active': active_strategy == 'multi_timeframe_trend'
            },
            {
                'id': 'support_resistance_breakout',
                'name': 'Support/Resistance Breakout',
                'description': 'Detects breakouts from key levels with volume confirmation',
                'active': active_strategy == 'support_resistance_breakout'
            }
        ]
        
        strategies = available_strategies
    
    return render_template('strategies.html', strategies=strategies)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists')
            return render_template('register.html')
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            
            # Log in the new user
            login_user(new_user)
            
            return redirect(url_for('dashboard'))
        except SQLAlchemyError as e:
            logger.error(f"Database error during registration: {e}")
            db.session.rollback()
            flash('An error occurred during registration')
    
    return render_template('register.html')


# API Routes
@app.route('/api/market_data', methods=['GET'])
def api_market_data():
    """API endpoint for market data"""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '15m')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'})
    
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    try:
        # Update symbol and timeframe if different
        if bot.symbol != symbol or bot.timeframe != timeframe:
            bot.symbol = symbol
            bot.timeframe = timeframe
            bot.update_candles()
        
        # Prepare candles data
        candles = []
        if bot.candles_df is not None:
            df = bot.candles_df.copy()
            for i, row in df.iterrows():
                candle = {
                    'timestamp': int(i.timestamp() * 1000) if hasattr(i, 'timestamp') else i,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                candles.append(candle)
        
        # Get ticker data
        ticker = None
        try:
            ticker_data = bot.get_ticker()
            if ticker_data:
                ticker = {
                    'last': float(ticker_data.get('last', 0)),
                    'bid': float(ticker_data.get('bid', 0)),
                    'ask': float(ticker_data.get('ask', 0)),
                    'volume': float(ticker_data.get('volume', 0)),
                    'change': float(ticker_data.get('percentage', 0)),
                    'high': float(ticker_data.get('high', 0)),
                    'low': float(ticker_data.get('low', 0))
                }
        except Exception as e:
            logger.error(f"Error fetching ticker data: {e}")
        
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': candles,
            'ticker': ticker,
            'last_update': int(time.time() * 1000)
        })
    
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({'error': str(e)})


@app.route('/api/performance', methods=['GET'])
def api_performance():
    """API endpoint for performance data"""
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    try:
        # Get performance data from state
        state = bot.state
        
        performance = {
            'total_trades': state.get('trades', {}).get('total', 0),
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        # Calculate win rate
        wins = state.get('trades', {}).get('wins', 0)
        if performance['total_trades'] > 0:
            performance['win_rate'] = (wins / performance['total_trades']) * 100
        
        # Get PnL and drawdown
        performance['total_pnl'] = state.get('performance', {}).get('pnl_percentage', 0.0)
        performance['max_drawdown'] = state.get('performance', {}).get('drawdown_max', 0.0)
        
        return jsonify(performance)
    
    except Exception as e:
        logger.error(f"Error fetching performance data: {e}")
        return jsonify({'error': str(e)})


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """API endpoint for configuration"""
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    if request.method == 'POST':
        try:
            # Get configuration data from request
            config_data = request.json
            
            if not config_data:
                return jsonify({'status': 'error', 'message': 'No configuration data provided'})
            
            # Update bot configuration
            for key, value in config_data.items():
                if isinstance(value, dict) and key in bot.config:
                    # Update nested dictionary
                    bot.config[key].update(value)
                else:
                    # Update top-level key
                    bot.config[key] = value
            
            # Save configuration to file
            with open(bot.config_file, 'w') as f:
                json.dump(bot.config, f, indent=2)
            
            # Reinitialize the bot with new configuration
            bot.initialize()
            
            return jsonify({'status': 'success', 'message': 'Configuration updated successfully'})
        
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return jsonify({'status': 'error', 'message': str(e)})
    
    else:
        # Return current configuration
        return jsonify(bot.config)


@app.route('/api/positions', methods=['GET'])
def api_positions():
    """API endpoint for positions data"""
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    try:
        # Update positions
        bot.update_positions()
        
        return jsonify({
            'positions': bot.current_positions,
            'last_update': int(time.time() * 1000)
        })
    
    except Exception as e:
        logger.error(f"Error fetching positions data: {e}")
        return jsonify({'error': str(e)})


@app.route('/api/close_position', methods=['POST'])
def api_close_position():
    """API endpoint to close a position"""
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    try:
        # Get position data from request
        position_data = request.json
        
        if not position_data:
            return jsonify({'status': 'error', 'message': 'No position data provided'})
        
        symbol = position_data.get('symbol')
        side = position_data.get('side')
        
        if not symbol or not side:
            return jsonify({'status': 'error', 'message': 'Symbol and side are required'})
        
        # Close position
        result = bot.close_position(side, symbol)
        
        if result:
            return jsonify({'status': 'success', 'message': f'Position closed for {symbol}'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to close position'})
    
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/start_bot', methods=['POST'])
def api_start_bot():
    """API endpoint to start the trading bot"""
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    try:
        # Start the bot
        bot.start()
        
        return jsonify({'status': 'success', 'message': 'Trading bot started'})
    
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/stop_bot', methods=['POST'])
def api_stop_bot():
    """API endpoint to stop the trading bot"""
    # Get bot instance
    bot = get_bot_instance()
    
    if not bot:
        return jsonify({'error': 'Trading bot not initialized'})
    
    try:
        # Stop the bot
        bot.stop()
        
        return jsonify({'status': 'success', 'message': 'Trading bot stopped'})
    
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}")
    return render_template('errors/500.html'), 500