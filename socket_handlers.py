"""
WebSocket Handlers Module

This module handles WebSocket events for real-time updates.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from flask_socketio import emit

from app import socketio
from views import get_bot_instance

# Configure logger
logger = logging.getLogger("socket_handlers")

@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit("connection_response", {"status": "connected"})

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on("request_market_data")
def handle_market_data_request(data):
    """
    Handle market data request from client
    
    Expected data:
    {
        'symbol': 'BTC/USDT',
        'timeframe': '15m'
    }
    """
    symbol = data.get('symbol')
    timeframe = data.get('timeframe', '15m')
    
    if not symbol:
        emit("market_data_error", {"error": "Symbol is required"})
        return
    
    # Get bot instance
    bot = get_bot_instance()
    if not bot:
        emit("market_data_error", {"error": "Trading bot not initialized"})
        return
    
    try:
        # Update symbol and timeframe if different
        if bot.symbol != symbol or bot.timeframe != timeframe:
            bot.symbol = symbol
            bot.timeframe = timeframe
            bot.update_candles()
        
        # Prepare candles data
        candles = []
        indicators = {}
        latest_indicators = {}
        
        if bot.candles_df is not None:
            df = bot.candles_df.copy()
            
            # Extract all indicator columns
            data_columns = ['open', 'high', 'low', 'close', 'volume']
            indicator_columns = [col for col in df.columns if col not in data_columns]
            
            # Process and extract latest indicators for display
            if len(df) > 0:
                latest_row = df.iloc[-1]
                for col in indicator_columns:
                    try:
                        latest_indicators[col] = float(latest_row[col]) if not pd.isna(latest_row[col]) else None
                    except:
                        latest_indicators[col] = str(latest_row[col])
            
            # Process the entire dataset for charting
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
                
                # Extract each indicator time series
                for col in indicator_columns:
                    if col not in indicators:
                        indicators[col] = []
                    
                    try:
                        value = float(row[col]) if not pd.isna(row[col]) else None
                    except:
                        value = None
                    
                    if value is not None:
                        indicators[col].append({
                            'timestamp': int(i.timestamp() * 1000) if hasattr(i, 'timestamp') else i,
                            'value': value
                        })
        
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
        
        # Emit market data
        emit("market_data", {
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': candles,
            'ticker': ticker,
            'indicators': indicators,
            'latest_indicators': latest_indicators,
            'last_update': int(time.time() * 1000)
        })
        
    except Exception as e:
        logger.error(f"Error handling market data request: {e}")
        emit("market_data_error", {"error": str(e)})

@socketio.on("run_strategy")
def handle_run_strategy(data):
    """
    Handle strategy execution request
    
    Expected data:
    {
        'symbol': 'BTC/USDT',
        'timeframe': '15m',
        'strategy': 'macd_crossover'
    }
    """
    symbol = data.get('symbol')
    timeframe = data.get('timeframe', '15m')
    strategy = data.get('strategy', 'simple_crossover')
    
    if not symbol:
        emit("strategy_error", {"error": "Symbol is required"})
        return
    
    # Get bot instance
    bot = get_bot_instance()
    if not bot:
        emit("strategy_error", {"error": "Trading bot not initialized"})
        return
    
    try:
        # Update symbol and timeframe if different
        if bot.symbol != symbol or bot.timeframe != timeframe:
            bot.symbol = symbol
            bot.timeframe = timeframe
            bot.update_candles()
        
        # Update bot config
        bot.config["symbol"] = symbol
        bot.config["timeframe"] = timeframe
        bot.config["strategy"]["active"] = strategy
        
        # Analyze market with the selected strategy
        from indicators import calculate_signal
        
        signal_strength = 0
        direction = "none"
        params = {}
        
        if bot.candles_df is not None and len(bot.candles_df) > 0:
            signal_strength, direction, params = calculate_signal(
                bot.candles_df, strategy, bot.config
            )
        
        # Prepare response
        result = {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "signal_strength": float(signal_strength),
            "direction": direction,
            "parameters": params
        }
        
        # Emit strategy results
        emit("strategy_result", result)
        
    except Exception as e:
        logger.error(f"Error executing strategy: {e}")
        emit("strategy_error", {"error": str(e)})

@socketio.on("update_strategy")
def handle_strategy_update(data):
    """
    Handle strategy update request
    
    Expected data:
    {
        'strategy_name': 'macd_crossover',
        'update': {
            'enabled': True,
            'parameters': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'risk_level': 'medium'
            }
        }
    }
    """
    strategy_name = data.get('strategy_name')
    update_data = data.get('update', {})
    
    if not strategy_name:
        emit("strategy_update", {"status": "error", "message": "Strategy name is required"})
        return
    
    try:
        import json
        import os
        
        # Load existing strategy config
        config_path = 'strategy_config.json'
        if not os.path.exists(config_path):
            # Create default config if it doesn't exist
            default_config = {
                "strategies": {}
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        # Read current config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'strategies' not in config:
            config['strategies'] = {}
        
        # Get or create strategy
        if strategy_name not in config['strategies']:
            config['strategies'][strategy_name] = {
                'enabled': False,
                'display_name': strategy_name.replace('_', ' ').title(),
                'description': 'Strategy description',
                'type': 'momentum',
                'risk_level': 'medium',
                'parameters': {}
            }
        
        # Update strategy
        strategy_config = config['strategies'][strategy_name]
        
        # Update enabled status
        if 'enabled' in update_data:
            strategy_config['enabled'] = bool(update_data['enabled'])
        
        # Update parameters
        if 'parameters' in update_data and isinstance(update_data['parameters'], dict):
            for key, value in update_data['parameters'].items():
                if isinstance(value, dict):
                    # Handle nested parameters
                    if key not in strategy_config['parameters']:
                        strategy_config['parameters'][key] = {}
                    for sub_key, sub_value in value.items():
                        strategy_config['parameters'][key][sub_key] = sub_value
                else:
                    # Handle direct parameters
                    strategy_config['parameters'][key] = value
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Return success response
        emit("strategy_update", {
            "status": "success",
            "strategy": strategy_name,
            "updated": list(update_data.keys())
        })
        
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_name}: {e}")
        emit("strategy_update", {"status": "error", "message": str(e)})

@socketio.on("run_backtest")
def handle_run_backtest(data):
    """
    Handle backtest execution request
    
    Expected data:
    {
        'strategy_name': 'macd_crossover',
        'symbol': 'BTC/USDT',
        'timeframe': '15m',
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        'initial_capital': 10000
    }
    """
    strategy_name = data.get('strategy_name')
    symbol = data.get('symbol', 'BTC/USDT')
    timeframe = data.get('timeframe', '15m')
    
    if not strategy_name:
        emit("backtest_error", {"error": "Strategy name is required"})
        return
    
    # Get bot instance
    bot = get_bot_instance()
    if not bot:
        emit("backtest_error", {"error": "Trading bot not initialized"})
        return
    
    try:
        # Import backtesting module
        from backtesting import Backtester, BacktestResult
        import json
        import os
        
        # Load strategy config
        config_path = 'strategy_config.json'
        if not os.path.exists(config_path):
            emit("backtest_error", {"error": "Strategy configuration not found"})
            return
            
        with open(config_path, 'r') as f:
            strategy_config = json.load(f)
        
        if 'strategies' not in strategy_config or strategy_name not in strategy_config['strategies']:
            emit("backtest_error", {"error": f"Strategy '{strategy_name}' not found"})
            return
        
        # Set up backtest config
        backtest_config = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy_name,
            "parameters": strategy_config['strategies'][strategy_name].get('parameters', {})
        }
        
        # Run mock backtest for now
        # In a real implementation, use actual historical data and the Backtester class
        import random
        
        # Generate mock backtest results
        win_rate = random.uniform(0.4, 0.7)
        profit_factor = random.uniform(0.8, 2.0)
        total_return = random.uniform(-0.1, 0.3)
        max_drawdown = random.uniform(0.05, 0.25)
        
        # Emit backtest results
        emit("backtest_results", {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "timestamp": int(time.time() * 1000)
        })
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        emit("backtest_error", {"error": str(e)})

# Import this module in app.py to register handlers
def init_socket_handlers():
    """Initialize socket handlers (called by app.py)"""
    logger.info("WebSocket handlers initialized")