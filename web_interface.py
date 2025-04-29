"""
Web Interface for Trading Bot

This module implements a Flask web interface for the trading bot
to display statistics, control the bot, and view trading history.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request, session

# Configure logger
logger = logging.getLogger("web_interface")

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SESSION_SECRET", "trading_bot_development_key")

# File paths
CONFIG_FILE = "config.json"
STATE_FILE = "bot_state.json"


def load_config() -> Dict:
    """Load configuration from file"""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading config: {e}")
        return {}


def load_state() -> Dict:
    """Load bot state from file"""
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading state: {e}")
        return {"positions": {}, "orders": {}, "trades": [], "last_update": 0}


def save_config(config: Dict) -> bool:
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


@app.route("/")
def index():
    """Render main dashboard page"""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Render main dashboard page"""
    config = load_config()
    state = load_state()
    
    # Format timestamps in trades for display
    for trade in state.get("trades", []):
        if "timestamp" in trade:
            trade["time"] = datetime.fromtimestamp(trade["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template(
        "dashboard.html",
        config=config,
        state=state,
        last_update=datetime.fromtimestamp(state.get("last_update", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S") if state.get("last_update", 0) > 0 else "Never"
    )


@app.route("/api/config", methods=["GET"])
def get_config():
    """API endpoint to get current configuration"""
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def update_config():
    """API endpoint to update configuration"""
    try:
        new_config = request.json
        if save_config(new_config):
            return jsonify({"status": "success", "message": "Configuration updated"})
        else:
            return jsonify({"status": "error", "message": "Failed to save configuration"}), 500
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/state", methods=["GET"])
def get_state():
    """API endpoint to get current bot state"""
    return jsonify(load_state())


@app.route("/api/performance", methods=["GET"])
def get_performance():
    """API endpoint to get trading performance statistics"""
    state = load_state()
    trades = state.get("trades", [])
    
    # Calculate performance metrics
    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum(trade.get("pnl", 0) for trade in trades)
    best_trade = max(trades, key=lambda x: x.get("pnl", 0))["pnl"] if trades else 0
    worst_trade = min(trades, key=lambda x: x.get("pnl", 0))["pnl"] if trades else 0
    
    # Calculate drawdown
    cumulative_pnl = 0
    max_cumulative_pnl = 0
    current_drawdown = 0
    max_drawdown = 0
    
    for trade in trades:
        pnl = trade.get("pnl", 0)
        cumulative_pnl += pnl
        
        if cumulative_pnl > max_cumulative_pnl:
            max_cumulative_pnl = cumulative_pnl
            current_drawdown = 0
        else:
            current_drawdown = max_cumulative_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, current_drawdown)
    
    return jsonify({
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "max_drawdown": max_drawdown,
        "current_positions": len(state.get("positions", {}))
    })

@app.route("/api/market_data", methods=["GET"])
def get_market_data():
    """API endpoint to get current market data"""
    try:
        from trading_bot import TradingBot
        
        # Create a temporary bot instance to fetch market data
        bot = TradingBot()
        
        # Load config and get selected symbol
        config = load_config()
        symbol = request.args.get("symbol", config.get("symbol", "BTC/USDT:USDT"))
        timeframe = request.args.get("timeframe", config.get("timeframe", "15m"))
        
        # Update symbol and timeframe
        bot.symbol = symbol
        bot.timeframe = timeframe
        
        # Fetch latest candles
        bot.update_candles()
        
        # Convert to dictionary for JSON serialization
        candles = []
        if bot.candles_df is not None:
            for index, row in bot.candles_df.tail(50).iterrows():  # Last 50 candles
                candle_data = {
                    "timestamp": index.timestamp() * 1000,  # Convert to JS timestamp
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                
                # Add indicators if available
                for indicator in ["rsi", "macd", "macd_signal", "macd_hist", 
                                 "bb_upper", "bb_middle", "bb_lower", 
                                 "ema_fast", "ema_slow", "atr"]:
                    if indicator in row and not pd.isna(row[indicator]):
                        candle_data[indicator] = float(row[indicator])
                
                candles.append(candle_data)
        
        # Get current market info
        ticker = bot.exchange.fetch_ticker(symbol)
        
        return jsonify({
            "candles": candles,
            "ticker": {
                "last": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "volume": ticker["volume"],
                "change": ticker["percentage"],
                "high": ticker["high"],
                "low": ticker["low"]
            },
            "last_update": datetime.now().timestamp() * 1000
        })
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({
            "error": str(e),
            "candles": [],
            "ticker": {}
        }), 500


if __name__ == "__main__":
    # This is used when running directly with Python
    # In production, use main.py instead
    app.run(host="0.0.0.0", port=5000, debug=True)